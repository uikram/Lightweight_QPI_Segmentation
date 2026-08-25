"""Standalone: two-pass calibrated dry mass from the 8-bit TIFFs.

  python calib_demo.py --tif-dir dataset/X_val --target-mch 30

Pass 1  per cell, sum (I - I_background) in 8-bit units, background anchored
        per image at the modal non-cell level (removes per-image offset).
Pass 2  solve one global scale s so the median day-0 cell equals target MCH,
        then report every day with that s.

The absolute pg values are CALIBRATED (anchored to fresh-cell MCH), not
measured. Only the relative change across storage is an experimental result.
"""
import argparse, csv, math, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

LAMBDA_UM, ALPHA, PIXEL_UM = 0.666, 0.20, 0.1441
K = PIXEL_UM**2 * LAMBDA_UM / (2*math.pi*ALPHA)      # pg per rad*px^2

ap = argparse.ArgumentParser()
ap.add_argument("--tif-dir", default="dataset/X_val")
ap.add_argument("--mask-dir", default="dataset/Y_val")
ap.add_argument("--trend", default="results/edge_sam_lora_r8/default_run/morphology_trends_rank_8.csv")
ap.add_argument("--target-mch", type=float, default=30.0)
ap.add_argument("--out", default="results/dry_mass_calibrated.csv")
ap.add_argument("--use-model", action="store_true",
                help="segment with EdgeSAM instead of reading --mask-dir (GT)")
ap.add_argument("--ckpt", default="results/edge_sam_lora_r8/checkpoints/best_model.pt")
ap.add_argument("--cfg", default="configs/edge_sam_lora.yaml")
a = ap.parse_args()

import tifffile
from scipy import ndimage

MODEL = DS = None
if a.use_model:
    import torch, yaml
    from datasets.qpi_dataset import QPIDataset
    from models import get_model

    print("=" * 68)
    print("MODEL SETUP")
    print("=" * 68)

    class _Cfg:
        num_classes = 5; pretrained = True; image_size = 1024
        lora_r = 8; lora_alpha = 8.0; insertion_strategy = "bottleneck"
    cfg = _Cfg()

    if Path(a.cfg).exists():
        y = yaml.safe_load(open(a.cfg)) or {}
        flat = {**y, **y.get("model", {}), **y.get("lora", {}),
                **y.get("dataset", {}), **y.get("training", {}), **y.get("data", {})}
        for k in ("num_classes", "image_size", "lora_r", "lora_alpha",
                  "insertion_strategy", "pretrained"):
            if k in flat:
                setattr(cfg, k, flat[k])
        print(f"[cfg] loaded {a.cfg}")
    else:
        print(f"[cfg] !! {a.cfg} NOT FOUND - using defaults, params will likely mismatch")
    for k in ("num_classes", "image_size", "lora_r", "lora_alpha",
              "insertion_strategy", "pretrained"):
        print(f"[cfg]   {k:20s} = {getattr(cfg, k)}")
    if not getattr(cfg, "pretrained", False):
        print("[cfg] !! pretrained=False -> weights/edge_sam_3x.pth will NOT be read")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dev] {dev}")
    MODEL = get_model("edge_sam", cfg).to(dev).eval()

    n_tot = sum(p_.numel() for p_ in MODEL.parameters())
    n_tra = sum(p_.numel() for p_ in MODEL.parameters() if p_.requires_grad)
    print(f"[arch] encoder = {MODEL.encoder.__class__.__name__}, "
          f"simple_decoder = {getattr(MODEL, 'use_simple_decoder', '?')}")
    print(f"[arch] params total {n_tot:,}  trainable {n_tra:,} ({100*n_tra/n_tot:.2f}%)")
    print(f"[arch] paper reports 534,205 / 6,004,333 (8.90%)")
    if abs(n_tot - 6004333) > 1000:
        d = n_tot - 6004333
        print(f"[arch] !! TOTAL MISMATCH vs paper by {d:+,}")
        if abs(d + 174885) < 1000:
            print("[arch]    note: 174,885 is exactly the parameter count of "
                  "simple_decoder (ConvT 256-128-64-32-16 + head).")
            print("[arch]    so this build is missing one decoder relative to the "
                  "trained model - check the branch reported above.")

    print(f"[ckpt] loading {a.ckpt}")
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    if isinstance(st, dict):
        print(f"[ckpt] top-level keys: {list(st.keys())[:8]}")
    sd = st.get("model_state", st.get("state_dict", st)) if isinstance(st, dict) else st
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    print(f"[ckpt] {len(sd)} tensors in file, {len(MODEL.state_dict())} in model")
    res = MODEL.load_state_dict(sd, strict=False)
    own = len(MODEL.state_dict()); matched = own - len(res.missing_keys)
    print(f"[ckpt] MATCHED {matched}/{own} ({100*matched/own:.1f}%)  "
          f"missing {len(res.missing_keys)}  unexpected {len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"[ckpt]   missing e.g.   : {res.missing_keys[:3]}")
    if res.unexpected_keys:
        print(f"[ckpt]   unexpected e.g.: {res.unexpected_keys[:3]}")
    if matched / max(own, 1) < 0.90:
        raise SystemExit("ABORT: checkpoint did not load. Fix cfg before trusting output.")

    DS = {}
    ds = QPIDataset(data_root="./dataset", split="val", augment=False)
    for i in range(len(ds)):
        smp = ds[i]; DS[smp.get("stem", str(i))] = smp
    print(f"[data] {len(DS)} val samples")

    # ---- functional check: a loaded model must beat a random one on Dice ----
    import numpy as _np
    dices, seen = [], 0
    with torch.no_grad():
        for stem_, smp in DS.items():
            if seen >= 12: break
            gt = (smp["mask"] > 0).numpy().astype(bool)
            if gt.sum() < 500: continue
            pr = MODEL(smp["phase"].unsqueeze(0).to(dev)).argmax(1).squeeze().cpu().numpy() > 0
            dices.append(2*(pr & gt).sum() / (pr.sum() + gt.sum() + 1e-9)); seen += 1
    md = float(_np.mean(dices)) if dices else 0.0
    print(f"[check] foreground Dice vs GT on {len(dices)} samples = {md:.3f}")
    if md < 0.40:
        raise SystemExit(f"ABORT: Dice {md:.3f} is near-random. Weights are not being used.")
    print(f"[check] OK - model is functional")
    print("=" * 68)

day_of = {r["stem"]: int(float(r["storage_day"]))
          for r in csv.DictReader(open(a.trend))} if Path(a.trend).exists() else {}

cells = []
for f in sorted(glob.glob(str(Path(a.tif_dir)/"*.tif"))):
    stem = Path(f).stem
    I = tifffile.imread(f).astype(np.float64)
    if a.use_model:
        import torch
        smp = DS.get(stem)
        if smp is None:
            continue
        with torch.no_grad():
            dev = next(MODEL.parameters()).device
            M = MODEL(smp["phase"].unsqueeze(0).to(dev)).argmax(1).squeeze().cpu().numpy()
        if M.shape != I.shape:                      # model runs at 256, phase at 768
            from PIL import Image as _Im
            M = np.array(_Im.fromarray(M.astype(np.uint8)).resize(
                (I.shape[1], I.shape[0]), _Im.NEAREST))
    else:
        mf = sorted(glob.glob(str(Path(a.mask_dir)/f"{stem}.*")))
        if not mf:
            continue
        M = tifffile.imread(mf[0])
    fg = M > 0
    hist = np.bincount(np.clip(I[~fg].astype(int), 0, 255), minlength=256)
    I_bg = float(np.argmax(hist))                      # modal background level
    lab, n = ndimage.label(fg)
    if n == 0:
        continue
    idx = np.arange(1, n+1)
    npix = ndimage.sum(np.ones_like(I), lab, idx)
    s_I  = ndimage.sum(I - I_bg, lab, idx)
    cls  = ndimage.maximum(M, lab, idx)          # predicted class per component
    try:                                          # crofton perimeter -> circularity
        from skimage.measure import regionprops
        peri = {r.label: r.perimeter for r in regionprops(lab)}
        per = np.array([peri.get(int(i), np.nan) for i in idx])
    except Exception:
        er = ndimage.binary_erosion(fg, ndimage.generate_binary_structure(2, 1))
        per = ndimage.sum((fg & ~er).astype(float), lab, idx)
    for np_, si, pe, cl in zip(npix, s_I, per, cls):
        area_um2 = np_ * PIXEL_UM**2
        if 15.0 <= area_um2 <= 200.0:
            circ = (4*math.pi*np_/(pe**2)) if pe and pe > 0 else float("nan")
            cells.append(dict(stem=stem, storage_day=day_of.get(stem, -1),
                              pred_class=int(cl),
                              area_um2=round(area_um2, 3),
                              circularity=round(float(circ), 4),
                              sum_I=si, npix=np_, I_bg=I_bg))

print(f"{len(cells)} cells from {len({c['stem'] for c in cells})} images")
bgs = [c["I_bg"] for c in cells]
print(f"background level across images: {min(bgs):.0f} .. {max(bgs):.0f} (8-bit)")

d0 = [c["sum_I"] for c in cells if c["storage_day"] == 0]
if not d0:
    raise SystemExit("no day-0 cells; cannot anchor")
s = a.target_mch / (float(np.median(d0)) * K)
print(f"\nscale s = {s:.6f} rad per 8-bit level  (0-255 spans {255*s:.2f} rad)")
print(f"anchored so median day-0 dry mass = {a.target_mch} pg\n")

by = defaultdict(list)
for c in cells:
    c["dry_mass_pg"] = c["sum_I"] * s * K
    by[c["storage_day"]].append(c)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cells[0].keys())); w.writeheader(); w.writerows(cells)

print(f"{'day':>4} {'n':>5} {'area um2':>13} {'circularity':>14} {'dry mass pg':>14}")
for d in sorted(k for k in by if k >= 0):
    g = by[d]
    m = np.array([x["dry_mass_pg"] for x in g]); ar = np.array([x["area_um2"] for x in g])
    ci = np.array([x["circularity"] for x in g]); ci = ci[~np.isnan(ci)]
    print(f"{d:>4} {len(g):>5} {np.mean(ar):>7.1f}+-{np.std(ar,ddof=1):>4.1f} "
          f"{np.mean(ci):>8.3f}+-{np.std(ci,ddof=1):>4.3f} "
          f"{np.mean(m):>8.1f}+-{np.std(m,ddof=1):>4.1f}")

print("\n% LaTeX rows for tab_8.tex")
for d in sorted(k for k in by if k >= 0):
    g = by[d]
    m = np.array([x["dry_mass_pg"] for x in g]); ar = np.array([x["area_um2"] for x in g])
    ci = np.array([x["circularity"] for x in g]); ci = ci[~np.isnan(ci)]
    print(f"{d:<3}& {len(g):<4}& ${np.mean(ar):.1f} \\pm {np.std(ar,ddof=1):.1f}$ "
          f"& ${np.mean(ci):.3f} \\pm {np.std(ci,ddof=1):.3f}$ "
          f"& ${np.mean(m):.1f} \\pm {np.std(m,ddof=1):.1f}$ \\\\")
print(f"\nwrote {a.out}")
print("NOTE: absolute pg are calibrated to MCH, not measured. Report the % column.")