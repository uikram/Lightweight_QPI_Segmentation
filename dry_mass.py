"""
Per-cell dry-mass extraction from the FLOAT phase reconstructions.

Why this exists
---------------
The original morphology pipeline summed pixel values of the 8-bit phase
images, so `opt_volume` was in arbitrary units x px^2, not rad x px^2, and the
`dry_mass` column was a verbatim copy of it.  This module redoes the
integration on the 32-bit float reconstructions (genuine radians) and applies
the physical calibration

        V_phi = sum_{i in Omega} phi_i * dx^2          [rad * um^2]
        m     = (lambda / (2 * pi * alpha)) * V_phi    [pg]

Calibration constants supplied by Seonghwan (DHM configuration used in-lab):
        lambda = 666 nm
        alpha  = 0.2 mL/g   (haemoglobin-dominated RBCs)
        dx     = 0.1441 um/px in the saved image

which gives 1 rad*px^2 = 0.011005 pg.

Note on absolute accuracy: dx was not measured on this specific acquisition,
so the absolute pg values carry a calibration uncertainty.  Because a wrong
dx^2 is a constant multiplier, it cancels in every *relative* comparison, so
the longitudinal trends and all significance statistics are unaffected.

Usage
-----
    python dry_mass.py                      # defaults below
    python dry_mass.py --data-root ./dataset --out results/dry_mass_cells.csv

Outputs
-------
    results/dry_mass_cells.csv   one row per detected cell
    results/dry_mass_by_day.csv  per-storage-day summary (mean +- SD)
    stdout                       LaTeX rows ready to paste into tab_8.tex
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Physical calibration
# --------------------------------------------------------------------------
LAMBDA_UM = 0.666      # illumination wavelength [um]
ALPHA     = 0.20       # specific refraction increment [mL/g] == [um^3/pg]
PIXEL_UM  = 0.1441     # effective pixel pitch in the saved image [um/px]

DA_UM2   = PIXEL_UM ** 2                          # um^2 per pixel
PREFAC   = LAMBDA_UM / (2.0 * math.pi * ALPHA)    # pg per (rad*um^2)
PG_PER_RAD_PX2 = DA_UM2 * PREFAC                  # 0.011005 pg per rad*px^2

# plausible single-RBC footprint, used only to drop specks and obvious clumps
MIN_AREA_UM2, MAX_AREA_UM2 = 15.0, 200.0

CLASS_NAME = {1: "discocyte", 2: "echinocyte", 3: "spherocyte", 4: "stomatocyte"}


# --------------------------------------------------------------------------
def _regions(mask_bin, phase):
    """Per-component area (px), perimeter (px) and integrated phase (rad*px)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask_bin)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    npix = ndimage.sum(np.ones_like(phase), lab, idx)
    sphi = ndimage.sum(phase, lab, idx)

    try:                                   # accurate crofton perimeter
        from skimage.measure import regionprops
        peri = {r.label: r.perimeter for r in regionprops(lab)}
        per = np.array([peri.get(int(i), np.nan) for i in idx])
    except Exception:                      # fallback: boundary pixel count
        er = ndimage.binary_erosion(mask_bin, ndimage.generate_binary_structure(2, 1))
        edge = mask_bin & ~er
        per = ndimage.sum(edge.astype(float), lab, idx)

    return list(zip(npix, per, sphi))


def _phase_from_dir(phase_dir, stem):
    """Load the float phase reconstruction for one patch, by stem.

    Looks for <stem>.npy / .npz / .tif / .tiff in phase_dir (recursively).
    Returns None if nothing matches, so the caller can fall back.
    """
    if phase_dir is None:
        return None
    for ext in (".npy", ".npz", ".tif", ".tiff"):
        hits = list(Path(phase_dir).rglob(f"{stem}{ext}"))
        if not hits:
            continue
        p = hits[0]
        if ext == ".npy":
            return np.load(p)
        if ext == ".npz":
            z = np.load(p)
            return z[list(z.keys())[0]]
        try:
            import tifffile
            return tifffile.imread(p)
        except ImportError:
            from PIL import Image
            return np.array(Image.open(p))
    return None


def _raw_phase(sample):
    """Return the UNNORMALISED phase in radians.

    The network is fed a standardised tensor, but the physics-aware loss is
    given the raw phase, so the dataset should expose it.  Key names differ
    between revisions, hence the search order.  If only the normalised tensor
    is available the function raises rather than silently integrating a
    z-scored array, which would be meaningless.
    """
    for key in ("phase_raw", "raw_phase", "phi_raw", "phi", "phase_rad"):
        if key in sample:
            return np.asarray(sample[key]).squeeze()
    ph = np.asarray(sample["phase"]).squeeze()
    if abs(float(ph.mean())) < 0.05 and 0.7 < float(ph.std()) < 1.4:
        raise KeyError(
            "Only a standardised phase tensor was found (mean~0, std~1). "
            "Integrating it would not give radians. Expose the raw phase from "
            "QPIDataset under one of: phase_raw / raw_phase / phi_raw / phi."
        )
    return ph


# --------------------------------------------------------------------------
def _load_checkpoint_verified(model, ckpt, dev):
    """Load weights and REFUSE to continue if they did not actually apply.

    strict=False silently ignores every key that does not match, so a config
    mismatch produces a randomly-initialised network that still runs and still
    emits plausible-looking masks.  This checks the overlap explicitly.
    """
    import torch
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = state.get("model_state", state.get("state_dict", state))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    own = set(model.state_dict().keys())
    matched = len(own) - len(res.missing_keys)
    print(f"checkpoint: {matched}/{len(own)} tensors matched, "
          f"{len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected")
    if res.missing_keys[:5]:
        print("  first missing :", res.missing_keys[:5])
    if res.unexpected_keys[:5]:
        print("  first unexpect:", res.unexpected_keys[:5])
    if matched / max(len(own), 1) < 0.90:
        raise RuntimeError(
            f"only {100*matched/len(own):.1f}% of weights loaded - the model built here "
            "does not match the trained one. Fix the config (lora_r, lora_alpha, "
            "insertion_strategy, image_size) before trusting any output.")
    return model


def export_dry_mass(data_root="./dataset",
                    ckpt="results/edge_sam_lora_r8/checkpoints/best_model.pt",
                    results_dir="results",
                    out_csv="results/dry_mass_cells.csv",
                    day_csv="results/dry_mass_by_day.csv",
                    erode_px=0,
                    phase_dir=None,
                    cfg_yaml="configs/edge_sam_lora.yaml"):
    import torch
    from datasets.qpi_dataset import QPIDataset
    from models import get_model
    from scipy import ndimage

    results_dir = Path(results_dir)

    class _Cfg:
        num_classes = 5; pretrained = False; image_size = 256
        lora_r = 8; lora_alpha = 8.0; insertion_strategy = "encoder_only"

    cfg = _Cfg()
    if cfg_yaml and Path(cfg_yaml).exists():          # prefer the TRAINING config
        try:
            import yaml
            y = yaml.safe_load(open(cfg_yaml))
            flat = {**y, **y.get("model", {}), **y.get("lora", {})}
            for k in ("num_classes", "image_size", "lora_r", "lora_alpha",
                      "insertion_strategy", "pretrained"):
                if k in flat:
                    setattr(cfg, k, flat[k])
            print(f"config from {cfg_yaml}: r={cfg.lora_r}, alpha={cfg.lora_alpha}, "
                  f"strategy={cfg.insertion_strategy}, size={cfg.image_size}")
        except Exception as e:
            print(f"could not read {cfg_yaml} ({e}); using defaults")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("edge_sam", cfg).to(dev).eval()
    _load_checkpoint_verified(model, ckpt, dev)
    ds = QPIDataset(data_root=data_root, split="val", augment=False)

    phase_dir = Path(phase_dir) if phase_dir else None
    if phase_dir:
        print(f"reading FLOAT phase from {phase_dir}")

    # stem -> storage day, taken from the existing morphology table
    day_of = {}
    trend = results_dir / "edge_sam_lora_r8/default_run/morphology_trends_rank_8.csv"
    if trend.exists():
        for r in csv.DictReader(open(trend)):
            day_of[r["stem"]] = int(float(r["storage_day"]))

    print(f"calibration: dx={PIXEL_UM} um/px, lambda={LAMBDA_UM*1000:.0f} nm, "
          f"alpha={ALPHA}  ->  1 rad*px^2 = {PG_PER_RAD_PX2:.6f} pg")

    rows, ranges = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            stem = s.get("stem", str(i))
            phase = _phase_from_dir(phase_dir, stem) if phase_dir else None
            if phase is None:
                phase = _raw_phase(s)
            phase = np.asarray(phase, dtype=np.float64).squeeze()
            ranges.append((phase.min(), phase.max()))

            logits = model(s["phase"].unsqueeze(0).to(dev))
            pred = logits.argmax(1).squeeze().cpu().numpy()

            for cls in (1, 2, 3, 4):
                m = pred == cls
                if not m.any():
                    continue
                if erode_px:
                    m = ndimage.binary_erosion(
                        m, ndimage.generate_binary_structure(2, 1), iterations=erode_px)
                for npix, per, sphi in _regions(m, phase):
                    a_um2 = npix * DA_UM2
                    if not (MIN_AREA_UM2 <= a_um2 <= MAX_AREA_UM2):
                        continue
                    circ = (4 * math.pi * npix / (per ** 2)) if per and per > 0 else float("nan")
                    rows.append(dict(
                        stem=stem, storage_day=day_of.get(stem, -1), pred_class=cls,
                        class_name=CLASS_NAME[cls],
                        area_px2=round(float(npix), 1),
                        area_um2=round(a_um2, 3),
                        circularity=round(float(circ), 4),
                        V_phi_rad_px2=round(float(sphi), 2),
                        V_phi_rad_um2=round(float(sphi) * DA_UM2, 4),
                        dry_mass_pg=round(float(sphi) * PG_PER_RAD_PX2, 4)))

    lo = min(r[0] for r in ranges); hi = max(r[1] for r in ranges)
    print(f"raw phase range across the split: {lo:.3f} .. {hi:.3f} rad")
    if hi > 50:
        print("  !! values above ~50 -> this is NOT radians, it is the 8-bit copy. Stop and fix.")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} cells -> {out_csv}")

    # ---- per-day summary -------------------------------------------------
    by = defaultdict(list)
    for r in rows:
        by[r["storage_day"]].append(r)
    with open(day_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["storage_day", "n_cells", "area_um2_mean", "area_um2_sd",
                    "circularity_mean", "circularity_sd",
                    "dry_mass_pg_mean", "dry_mass_pg_sd", "dominant_class"])
        for d in sorted(k for k in by if k >= 0):
            g = by[d]
            a = np.array([x["area_um2"] for x in g])
            c = np.array([x["circularity"] for x in g])
            m = np.array([x["dry_mass_pg"] for x in g])
            names = [x["class_name"] for x in g]
            dom = max(set(names), key=names.count)
            w.writerow([d, len(g), f"{a.mean():.1f}", f"{a.std(ddof=1):.1f}",
                        f"{c.mean():.3f}", f"{c.std(ddof=1):.3f}",
                        f"{m.mean():.1f}", f"{m.std(ddof=1):.1f}", dom])
    print(f"wrote per-day summary -> {day_csv}\n")

    # ---- LaTeX rows for tab_8.tex ---------------------------------------
    print("% paste into tab_8.tex (columns: Day, Area, Circularity, Dry mass, Classes)")
    for d in sorted(k for k in by if k >= 0):
        g = by[d]
        a = np.array([x["area_um2"] for x in g]); c = np.array([x["circularity"] for x in g])
        m = np.array([x["dry_mass_pg"] for x in g])
        names = [x["class_name"] for x in g]
        comp = ", ".join(f"{n[:6].capitalize()}.\\ {names.count(n)}"
                         for n in sorted(set(names), key=names.count, reverse=True))
        print(f"{d:<3}& ${a.mean():.1f} \\pm {a.std(ddof=1):.1f}$ "
              f"& ${c.mean():.3f} \\pm {c.std(ddof=1):.3f}$ "
              f"& ${m.mean():.1f} \\pm {m.std(ddof=1):.1f}$ & {comp} \\\\")
    all_m = np.array([r["dry_mass_pg"] for r in rows])
    print(f"\n% overall: median {np.median(all_m):.1f} pg, "
          f"IQR {np.percentile(all_m,25):.1f}-{np.percentile(all_m,75):.1f}, n={len(all_m)}")
    print("% fresh-RBC reference (MCH): 27-33 pg")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./dataset")
    ap.add_argument("--ckpt", default="results/edge_sam_lora_r8/checkpoints/best_model.pt")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/dry_mass_cells.csv")
    ap.add_argument("--day-out", default="results/dry_mass_by_day.csv")
    ap.add_argument("--erode-px", type=int, default=0,
                    help="optional mask erosion in pixels (sensitivity check)")
    ap.add_argument("--phase-dir", default=None,
                    help="directory holding the FLOAT phase reconstructions "
                         "(<stem>.npy/.tif); required if the dataset serves 8-bit")
    ap.add_argument("--cfg", default="configs/edge_sam_lora.yaml",
                    help="training config, used so the model matches the checkpoint")
    a = ap.parse_args()
    export_dry_mass(a.data_root, a.ckpt, a.results_dir, a.out, a.day_out,
                    a.erode_px, a.phase_dir, a.cfg)