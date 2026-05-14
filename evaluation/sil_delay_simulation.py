"""
Software-in-the-Loop (SiL) Simulation 
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# FIX-A: Runtime detection of the correct integration function.
# np.trapz was removed in NumPy 2.0 and replaced with np.trapezoid.
# This one-time detection works on any NumPy version >= 1.x.
try:
    _trapz = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz       # NumPy < 2.0

# =============================================================================
#                              CONFIGURATION BLOCK
# =============================================================================

# --- Output Settings ---
OUTPUT_DIR = '../stability_result'

# --- Plant and Controller Parameters ---
M       = 1.2           # Mass (kg) [de Leva 1996]
f_n     = 3.0           # Natural frequency (Hz) [Gallego 2010]
zeta    = 0.707         # Damping ratio [Ogata 2010, Butterworth]
Kp      = 1200.0        # Proportional gain
Ki      = 1000.0        # Integral gain
Kd      = 5.0           # Derivative gain

# --- Simulation Settings ---
SIM_DT              = 0.0005    # Integration timestep (s) -> 0.5 ms / 2 kHz
SIM_T_TOTAL         = 3.0       # Total simulation time (s)
SIM_VISION_HZ       = 30.0      # Vision system update rate (Hz)
SIM_BAND_TOLERANCE  = 0.05      # Tolerance band for settling time (±5%)
SIM_BAND_WINDOW_S   = 0.050     # Sustained window required for settling (s)

# --- Stability Boundary Search Settings ---
SEARCH_TAU_LO       = 0.001     # Lower bound for tau search (s)
SEARCH_TAU_HI       = 0.200     # Upper bound for tau search (s)
SEARCH_ITERATIONS   = 60        # Binary search iterations

# --- Clean SiL Sweep Settings ---
SWEEP_DELAY_VALUES_MS = [10, 15, 20, 30, 40, 50]  # Delays to simulate (ms)

# --- WCL (Worst-Case Latency) Analysis Settings ---
T_JITTER   = 1.0   # OS scheduling jitter margin (ms)
T_BUFFER   = 2.0   # Deployment safety buffer (ms)
WCL_TARGET = 17.92 # WCL limit to evaluate against (ms)
LATENCY_DATA = {
    'CLIP Baseline':          {'p99': 14.82, 'mean': 14.24},
    'LoRA-CLIP (merged)':     {'p99': 14.92, 'mean': 14.34},
    'LoRA-CLIP (unmerged)':   {'p99': 24.01, 'mean': 22.87},
    'Frozen Prefix-LM (E2E)': {'p99': 439.02, 'mean': 430.37},
}

# --- Parametric Robustness Heatmap Settings ---
HEATMAP_GRID_SIZE = 40          # Resolution of the parameter grid
HEATMAP_M_RANGE   = (0.5, 1.5)  # Sweep multiplier bounds for Mass (M)
HEATMAP_K_RANGE   = (0.5, 1.5)  # Sweep multiplier bounds for Stiffness (K)

# --- Noise Sensitivity Sweep Settings ---
NOISE_N_SEEDS                  = 20
NOISE_LEVELS                   = [0.01, 0.02, 0.05, 0.10]
NOISE_IAE_THRESHOLD_MULTIPLIER = 3.0  # Multiplier over clean IAE to mark divergence
NOISE_DEFAULT_SEED             = 42

# --- Plotting Settings ---
COLORS_DELAY         = ['#1F77B4', '#00D2D3', '#10AC84', '#FF9F43', '#EE5253', '#833471']
COLORS_NOISE         = ['#1F77B4', '#10AC84', '#FF9F43', '#EE5253']
PLOT_T_END_FULL      = 3.0    # X-axis limit for full step response plot
PLOT_T_END_TRANSIENT = 0.8    # X-axis limit for transient step response plot
PLOT_CLIP_MIN        = -2     # Y-axis data clipping minimum
PLOT_CLIP_MAX        = 12     # Y-axis data clipping maximum
PLOT_YMIN            = -0.2   # Y-axis plot minimum
PLOT_YMAX_FULL       = 2.0    # Y-axis plot maximum for full step response
PLOT_YMAX_TRANS      = 3.0    # Y-axis plot maximum for transient step response

# =============================================================================
#                                SETUP & DERIVATIONS
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")
print(f"NumPy version: {np.__version__}  |  Integration function: {_trapz.__name__}\n")

json_export_data = {}

# Derived plant parameters
omega_n = 2 * np.pi * f_n
K       = M * omega_n**2
B       = 2 * zeta * np.sqrt(M * K)

print("=" * 58)
print("PLANT PARAMETERS")
print(f"  M       = {M}    kg")
print(f"  f_n     = {f_n}   Hz")
print(f"  zeta    = {zeta}")
print(f"  K       = {K:.4f} N/m")
print(f"  B       = {B:.4f} N·s/m")
print(f"  PID:    Kp={Kp}  Ki={Ki}  Kd={Kd}")
print("=" * 58)

# ── 2. VERIFY DELAY-FREE STABILITY ───────────────────────────────────────────
p0 = [M, B + Kd, K + Kp, Ki]
roots = np.roots(p0)
baseline_stable = all(r.real < 0 for r in roots)
print(f"\nDelay-free roots: {[f'{r:.4f}' for r in roots]}")
print(f"All in LHP: {baseline_stable}")
assert baseline_stable, "Baseline system not stable — check PID gains."

# ── 3. PADÉ COEFFICIENTS ──────────────────────────────────────────────────────
def poly_coeffs(tau, m=M, k=K, b=B):
    a4 = m * tau / 2
    a3 = m + (b - Kd) * tau / 2
    a2 = b + Kd + (k - Kp) * tau / 2
    a1 = k + Kp - Ki * tau / 2
    a0 = Ki
    return [a4, a3, a2, a1, a0]

def is_stable_roots(tau, m=M, k=K, b=B):
    coeffs = poly_coeffs(tau, m, k, b)
    if coeffs[0] <= 0:
        return False
    return all(r.real < 0 for r in np.roots(coeffs))

def find_tau_max(m_val=M, k_val=K, b_val=B):
    tau_lo, tau_hi = SEARCH_TAU_LO, SEARCH_TAU_HI
    for _ in range(SEARCH_ITERATIONS):
        tau_mid = (tau_lo + tau_hi) / 2
        if is_stable_roots(tau_mid, m_val, k_val, b_val):
            tau_lo = tau_mid
        else:
            tau_hi = tau_mid
    return tau_lo

# ── 4. NOMINAL TAU_MAX & ROUTH ARRAY ─────────────────────────────────────────
tau_max    = find_tau_max()
tau_max_ms = tau_max * 1000
print(f"\nNominal tau_max = {tau_max_ms:.4f} ms")
json_export_data["tau_max_ms"] = float(tau_max_ms)

a4, a3, a2, a1, a0 = poly_coeffs(tau_max)
b1 = (a3 * a2 - a4 * a1) / a3
c1 = (b1 * a1 - a3 * a0) / b1
json_export_data["routh_first_column"] = {
    "s4": float(a4), "s3": float(a3),
    "s2": float(b1), "s1": float(c1), "s0": float(a0)
}
print(f"Routh s^1 at boundary: {c1:.2e}  (should approach 0)")

# ── 5. SiL SIMULATION CORE ───────────────────────────────────────────────────

def run_sil(tau_ms, noise_sigma=0.0, noise_seed=NOISE_DEFAULT_SEED, noise_on_ref=True):
    dt           = SIM_DT
    T_total      = SIM_T_TOTAL
    t            = np.arange(0, T_total, dt)
    N            = len(t)
    x_ref_s      = np.ones(N)

    delay_steps     = int((tau_ms / 1000.0) / dt)
    vision_hz       = SIM_VISION_HZ
    steps_per_frame = int((1.0 / vision_hz) / dt)

    rng       = np.random.default_rng(noise_seed)  # isolated RNG, no global state
    noise_obs = np.zeros(N)
    noise_ref = np.zeros(N)

    if noise_sigma > 0:
        for i in range(0, N, steps_per_frame):
            noise_obs[i:i + steps_per_frame] = rng.normal(0, noise_sigma)
            if noise_on_ref:
                noise_ref[i:i + steps_per_frame] = rng.normal(0, noise_sigma)

    x        = np.zeros(N)
    xd       = np.zeros(N)
    integral = 0.0
    prev_x   = 0.0  # DoM: track delayed observed position

    for i in range(1, N):
        delayed_i   = max(0, i - delay_steps)
        observation = x[delayed_i] + noise_obs[i]
        reference   = x_ref_s[i]  + noise_ref[i]

        error     = reference - observation
        integral += error * dt

        deriv_meas = -(x[delayed_i] - prev_x) / dt  # DoM: continuous at t=0
        prev_x     = x[delayed_i]

        u = np.clip(Kp * error + Ki * integral + Kd * deriv_meas, -1e5, 1e5)

        xdd   = (u - B * xd[i - 1] - K * x[i - 1]) / M
        xd[i] = xd[i - 1] + xdd * dt
        x[i]  = x[i - 1] + xd[i - 1] * dt

    e   = x_ref_s - x
    IAE = float(_trapz(np.abs(e), t))   # FIX-A: version-safe integration
    ISE = float(_trapz(e**2, t))
    OS  = float((np.max(x) - 1.0) * 100.0)

    window  = int(SIM_BAND_WINDOW_S / dt)  
    Ts      = None
    in_band = np.abs(x - 1.0) <= SIM_BAND_TOLERANCE
    for i in range(N - window):
        if np.all(in_band[i: i + window]):
            Ts = float(t[i])
            break

    return dict(x=x, t=t, IAE=IAE, ISE=ISE, OS=OS, Ts=Ts,
                stable=(tau_ms / 1000.0) < tau_max)

# ── 6. CLEAN SiL SWEEP ───────────────────────────────────────────────────────
print("\nRUNNING CLEAN SiL SIMULATION...")
results = {}
json_export_data["sil_simulation"] = {}

print(f"\n{'tau(ms)':<10} {'IAE':<10} {'OS(%)':<12} {'ISE':<12} {'Ts(s)':<12} Status")
print("-" * 64)

for tau_ms in SWEEP_DELAY_VALUES_MS:
    r = run_sil(tau_ms, noise_sigma=0.0)
    results[tau_ms] = r
    ts_str = f"{r['Ts']:.3f}" if r['Ts'] is not None else "N/A"
    status = "STABLE" if r['stable'] else "UNSTABLE"
    print(f"{tau_ms:<10} {r['IAE']:<10.4f} {r['OS']:<12.2f} {r['ISE']:<12.4f} "
          f"{ts_str:<12} {status}")
    json_export_data["sil_simulation"][f"{tau_ms}ms"] = {
        "IAE":    round(r['IAE'], 4),
        "OS_pct": round(r['OS'],  2),
        "ISE":    round(r['ISE'], 4),
        "Ts_s":   round(r['Ts'],  3) if r['Ts'] is not None else "N/A",
        "stable": bool(r['stable'])
    }

# ── 7. WCL ANALYSIS ──────────────────────────────────────────────────────────
json_export_data["worst_case_latency"] = {}
print(f"\n{'Model':<32} {'p99':>8} {'WCL':>8} {'%tau_max':>10} {'Margin':>10}  Safe?")
print("-" * 76)
for model, d in LATENCY_DATA.items():
    wcl    = d['p99'] + T_JITTER + T_BUFFER
    pct    = wcl / tau_max_ms * 100
    margin = tau_max_ms - wcl
    safe   = wcl < tau_max_ms
    print(f"{model:<32} {d['p99']:>8.2f} {wcl:>8.2f} {pct:>9.1f}% "
          f"{margin:>10.2f}  {'SAFE' if safe else 'UNSAFE'}")
    json_export_data["worst_case_latency"][model] = {
        "p99_ms":        float(d['p99']),
        "wcl_ms":        wcl,
        "ratio_percent": round(pct, 1),
        "margin_ms":     margin,
        "safe":          bool(safe)
    }

# ── 8. PARAMETRIC ROBUSTNESS HEATMAP ─────────────────────────────────────────
print("\nCOMPUTING M vs K STABILITY HEATMAP...")
M_vals = np.linspace(M * HEATMAP_M_RANGE[0], M * HEATMAP_M_RANGE[1], HEATMAP_GRID_SIZE)
K_vals = np.linspace(K * HEATMAP_K_RANGE[0], K * HEATMAP_K_RANGE[1], HEATMAP_GRID_SIZE)
X_M, Y_K = np.meshgrid(M_vals, K_vals)
Z_tau = np.zeros((HEATMAP_GRID_SIZE, HEATMAP_GRID_SIZE))
min_tau_found = float('inf')
min_tau_m     = M
min_tau_k     = K

for i in range(HEATMAP_GRID_SIZE):
    for j in range(HEATMAP_GRID_SIZE):
        m_c = X_M[i, j]
        k_c = Y_K[i, j]
        b_c = 2 * zeta * np.sqrt(m_c * k_c)
        val = find_tau_max(m_c, k_c, b_c) * 1000.0
        Z_tau[i, j] = val
        if val < min_tau_found:
            min_tau_found = val
            min_tau_m     = m_c
            min_tau_k     = k_c

safety_margin_ms = min_tau_found - WCL_TARGET

# FIX-B: Export corner location to JSON for reproducibility
json_export_data["parametric_sweep"] = {
    "min_tau_max_ms":    round(float(min_tau_found), 2),
    "min_tau_at_M_kg":   round(float(min_tau_m), 3),
    "min_tau_at_K_Npm":  round(float(min_tau_k), 2),
    "WCL_target_ms":     WCL_TARGET,
    "safety_margin_ms":  round(float(safety_margin_ms), 2)
}
print(f"  Min tau_max: {min_tau_found:.2f} ms  at M={min_tau_m:.2f} kg, K={min_tau_k:.0f} N/m")
print(f"  Safety margin over WCL ({WCL_TARGET} ms): {safety_margin_ms:.2f} ms")

# ── 9. NOISE SENSITIVITY SWEEP ────────────────────────────────────────────────
print("\nRUNNING NOISE SENSITIVITY SWEEP...")

clean_r   = run_sil(WCL_TARGET, noise_sigma=0.0)
clean_IAE = clean_r['IAE']

STABILITY_IAE_THRESHOLD = NOISE_IAE_THRESHOLD_MULTIPLIER * clean_IAE

noisy_results = {}
json_export_data[f"noise_sweep_wcl_{WCL_TARGET}ms"] = {
    "clean_IAE":               round(float(clean_IAE), 4),
    "N_seeds":                 NOISE_N_SEEDS,
    "stability_iae_threshold": round(float(STABILITY_IAE_THRESHOLD), 4),
    "note": (
        "Noise injected independently into both observation (delayed feedback) "
        "and reference (vision-derived target) channels. Independent N(0,sigma) "
        "draws on both channels produce effective error noise of sigma*sqrt(2). "
        "Sigma range is bracketed by LoRA-CLIP empirical latency std (~6.4% of mean). "
        "Plotted time-domain trajectories are mean-representative seeds (closest IAE "
        "to mean across N_seeds). Bar chart shows full mean +/- std statistics."
    )
}

print(f"\n{'sigma':>8} {'eff. sigma':>12} {'mean IAE':>12} "
      f"{'std IAE':>10} {'deg%':>8}  Status")
print("-" * 66)

for sigma in NOISE_LEVELS:
    seed_IAEs = []
    seed_xs   = []
    for seed in range(NOISE_N_SEEDS):
        r = run_sil(WCL_TARGET, noise_sigma=sigma, noise_seed=seed, noise_on_ref=True)
        seed_IAEs.append(r['IAE'])
        seed_xs.append(r['x'])

    mean_IAE = float(np.mean(seed_IAEs))
    std_IAE  = float(np.std(seed_IAEs))
    max_IAE  = float(np.max(seed_IAEs))
    pct_lbl  = int(sigma * 100)
    deg_pct  = (mean_IAE - clean_IAE) / clean_IAE * 100
    eff_sigma = sigma * np.sqrt(2)

    bounded = mean_IAE < STABILITY_IAE_THRESHOLD
    status_str = "BOUNDED" if bounded else "DIVERGENT"

    best_idx = int(np.argmin(np.abs(np.array(seed_IAEs) - mean_IAE)))
    noisy_results[sigma] = {
        'x_repr':   seed_xs[best_idx],
        'mean_IAE': mean_IAE,
        'std_IAE':  std_IAE,
        'max_IAE':  max_IAE,
        'deg_pct':  deg_pct,
        'bounded':  bounded
    }

    print(f"  {pct_lbl:>3}%  {eff_sigma*100:>10.1f}%  {mean_IAE:>12.4f} "
          f"{std_IAE:>10.4f} {deg_pct:>7.1f}%  {status_str}")

    json_export_data[f"noise_sweep_wcl_{WCL_TARGET}ms"][f"sigma_{pct_lbl}pct"] = {
        "nominal_sigma":    sigma,
        "effective_sigma":  round(eff_sigma, 4),
        "mean_IAE":         round(mean_IAE, 4),
        "std_IAE":          round(std_IAE,  4),
        "max_IAE":          round(max_IAE,  4),
        "degradation_pct":  round(deg_pct, 1),
        "status":           status_str,
        "bounded":          bool(bounded)
    }

# ── 10. FIGURES ───────────────────────────────────────────────────────────────
print("\nGENERATING PLOTS...")

# ── Fig 7 & 8: Step response (full 3s + transient 0.8s) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Closed-Loop Step Response: PID-Controlled MSD with Perception Delay',
             fontsize=12, fontweight='bold')
for ax, t_end in zip(axes, [PLOT_T_END_FULL, PLOT_T_END_TRANSIENT]):
    t_arr = results[SWEEP_DELAY_VALUES_MS[0]]['t']
    mask  = t_arr <= t_end
    for tau_ms, color in zip(SWEEP_DELAY_VALUES_MS, COLORS_DELAY):
        r = results[tau_ms]
        ax.plot(t_arr[mask], np.clip(r['x'][mask], PLOT_CLIP_MIN, PLOT_CLIP_MAX), color=color,
                ls='--' if not r['stable'] else '-', lw=2,
                label=fr'$\tau = {tau_ms}\ \mathrm{{ms}}$' + (' [UNSTABLE]' if not r['stable'] else ''))
    ax.axhline(1.0,  color='black', lw=0.8, ls=':', alpha=0.7, label='Reference')
    ax.axhline(1.0 + SIM_BAND_TOLERANCE, color='gray',  lw=0.5, ls=':', alpha=0.4)
    ax.axhline(1.0 - SIM_BAND_TOLERANCE, color='gray',  lw=0.5, ls=':', alpha=0.4)
    ax.set_xlabel('Time (s)'); ax.set_ylabel(r'Position $x(t)$')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, t_end)
    ax.set_ylim(PLOT_YMIN, PLOT_YMAX_FULL if t_end > 1.0 else PLOT_YMAX_TRANS)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_7_8_step_response.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_7_8_step_response.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("✓ fig_7_8_step_response")

# ── Fig 10: 4-panel degradation curves ─────────────────────────────────────
tau_arr = SWEEP_DELAY_VALUES_MS
IAE_arr = [results[t]['IAE'] for t in tau_arr]
OS_arr  = [results[t]['OS']  for t in tau_arr]
ISE_arr = [results[t]['ISE'] for t in tau_arr]
Ts_arr  = [results[t]['Ts']  if results[t]['Ts'] is not None else np.nan for t in tau_arr]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('Latency–Stability Degradation Curves (SiL Simulation)',
             fontsize=12, fontweight='bold')
for vals, title, ylabel, ax in [
    (IAE_arr, 'Tracking Error (IAE)',     'IAE',    axes[0, 0]),
    (OS_arr,  'Overshoot (%)',            'OS (%)', axes[0, 1]),
    (ISE_arr, 'Oscillation Energy (ISE)', 'ISE',    axes[1, 0]),
    (Ts_arr,  'Settling Time (s)',        'Ts (s)', axes[1, 1])
]:
    ax.plot(tau_arr, vals, 'o-', color='#1A237E', lw=2.2, markersize=8,
            markerfacecolor='white', markeredgewidth=2.5, zorder=4)
    ax.axvspan(tau_max_ms, 55, alpha=0.10, color='red', zorder=1,
               label='Unstable region')
    merged_wcl = LATENCY_DATA['LoRA-CLIP (merged)']['p99'] + T_JITTER + T_BUFFER
    unmerged_wcl = LATENCY_DATA['LoRA-CLIP (unmerged)']['p99'] + T_JITTER + T_BUFFER

    ax.axvline(tau_max_ms, color='red', lw=2.0, ls='-', zorder=3,
               label=fr'$\tau_{{\max}} = {tau_max_ms:.2f}\ \mathrm{{ms}}$')
    ax.axvline(merged_wcl, color='purple', lw=1.5, ls='--', zorder=3,
               label=f"LoRA merged WCL ({merged_wcl:.2f} ms)")
    ax.axvline(unmerged_wcl, color='green',  lw=1.5, ls='--', zorder=3,
               label=f"LoRA unmerged WCL ({unmerged_wcl:.2f} ms)")
    ax.set_xlabel(r'Perception Delay $\tau$ $\mathrm{ms}$(ms)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(7, 53)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_10_degradation.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_10_degradation.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("✓ fig_10_degradation")

# ── Fig 11: Parametric Heatmap ────────────
fig, ax = plt.subplots(figsize=(8, 6))

levels = np.linspace(WCL_TARGET, Z_tau.max(), 40)
cmap = plt.get_cmap('viridis_r').copy()
cmap.set_under('lightcoral') 

cp = ax.contourf(X_M, Y_K, Z_tau, levels=levels, cmap=cmap, extend='min', alpha=0.9)
cbar = fig.colorbar(cp, ax=ax, pad=0.02)
# FIX 3: Standardized Units
cbar.set_label(r'Critical Delay Margin $\tau_{\max}$ $\mathrm{ms}$', fontsize=11, weight='bold')

# Danger Zone Hatching (Darker and more prominent)
ax.contourf(X_M, Y_K, Z_tau, levels=[-np.inf, WCL_TARGET], colors=['none'], hatches=['////'], alpha=0.8, zorder=3)

# Scale-invariant Worst-Case Boundary
z_range = Z_tau.max() - min_tau_found
epsilon = 0.02 * z_range if z_range > 0 else 0.5
ax.contour(X_M, Y_K, Z_tau, levels=[min_tau_found + epsilon], colors='black', linestyles=':', linewidths=1.5, zorder=4)

contours = ax.contour(X_M, Y_K, Z_tau, levels=8, colors='white', alpha=0.3, linewidths=0.8, zorder=4)
ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f ms')

ax.plot(M, K, marker='*', color='white', markersize=14, markeredgecolor='black', markeredgewidth=1.0, 
        label='Nominal Configuration', zorder=6)
# FIX 4: Unmistakable Worst-Case Marker
ax.plot(min_tau_m, min_tau_k, marker='v', color='red', markersize=12, markeredgecolor='black', markeredgewidth=1.5, 
        label='Worst-Case (Domain Minimum)', zorder=6)

if Z_tau.min() <= WCL_TARGET <= Z_tau.max():
    wcl_line = ax.contour(X_M, Y_K, Z_tau, levels=[WCL_TARGET], colors='red', linewidths=3.0, linestyles='--', zorder=5)
    ax.clabel(wcl_line, fmt='FAILURE THRESHOLD', fontsize=9, colors='red')

ax.text(0.03, 0.04, r'Stable if $\tau_{\max} > WCL_{\mathrm{target}}$',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round,pad=0.3'), zorder=7)

ax.set_title('Parametric Robustness of Delay Margin', fontweight='bold', fontsize=12)
ax.set_xlabel('Arm Mass M (kg)', fontsize=11)
ax.set_ylabel('Arm Stiffness K $\mathrm{N/m}$', fontsize=11)

ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black').set_zorder(7)
ax.grid(True, alpha=0.15, color='black')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_heatmap.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11_heatmap.pdf'), format='pdf', bbox_inches='tight')
plt.close()

import seaborn as sns
cb_colors = sns.color_palette("colorblind")

# ── Fig 11b: 2D Cross-Section Slices ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

m_line = np.linspace(M * HEATMAP_M_RANGE[0], M * HEATMAP_M_RANGE[1], 100)
k_line = np.linspace(K * HEATMAP_K_RANGE[0], K * HEATMAP_K_RANGE[1], 100)

tau_vary_m_nom = [find_tau_max(m_val, K, 2 * zeta * np.sqrt(m_val * K)) * 1000 for m_val in m_line]
tau_vary_m_wc  = [find_tau_max(m_val, min_tau_k, 2 * zeta * np.sqrt(m_val * min_tau_k)) * 1000 for m_val in m_line]

tau_vary_k_nom = [find_tau_max(M, k_val, 2 * zeta * np.sqrt(M * k_val)) * 1000 for k_val in k_line]
tau_vary_k_wc  = [find_tau_max(min_tau_m, k_val, 2 * zeta * np.sqrt(min_tau_m * k_val)) * 1000 for k_val in k_line]

# ---- Left Plot: Varying Mass ----
ax1.plot(m_line, tau_vary_m_nom, color=cb_colors[0], lw=2.5, label='Nominal (Fixed K)', zorder=3)
ax1.plot(m_line, tau_vary_m_wc, color=cb_colors[3], lw=2.5, ls='--', label='Worst-Case (Fixed K)', zorder=3)
ax1.axhline(WCL_TARGET, color='red', lw=2, label='WCL Target', zorder=2)
ax1.fill_between(m_line, 0, WCL_TARGET, color='none', edgecolor='red', hatch='////', alpha=0.5, zorder=1)

ax1.plot(M, tau_max_ms, marker='*', color=cb_colors[0], markersize=14, markeredgecolor='black', markeredgewidth=1.0, zorder=6)
ax1.plot(min_tau_m, min_tau_found, marker='v', color=cb_colors[3], markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=6)

# FIX 2: Stronger Titles
ax1.set_title('Worst-Case Delay Margin vs. Arm Mass', fontweight='bold', fontsize=11)
ax1.set_xlabel('Arm Mass M (kg)', fontsize=11)
ax.set_ylabel(r'Critical Delay Margin $\tau_{\max}$ $\mathrm{ms}$', fontsize=11)
ax1.set_xlim(m_line.min(), m_line.max())
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='upper right').set_zorder(7)

# ---- Right Plot: Varying Stiffness ----
ax2.plot(k_line, tau_vary_k_nom, color=cb_colors[2], lw=2.5, label='Nominal (Fixed M)', zorder=3)
ax2.plot(k_line, tau_vary_k_wc, color=cb_colors[1], lw=2.5, ls='--', label='Worst-Case (Fixed M)', zorder=3)
ax2.axhline(WCL_TARGET, color='red', lw=2, zorder=2)
ax2.fill_between(k_line, 0, WCL_TARGET, color='none', edgecolor='red', hatch='////', alpha=0.5, zorder=1)

ax2.plot(K, tau_max_ms, marker='*', color=cb_colors[2], markersize=14, markeredgecolor='black', markeredgewidth=1.0, zorder=6)
ax2.plot(min_tau_k, min_tau_found, marker='v', color=cb_colors[1], markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=6)

ax2.set_title('Worst-Case Delay Margin vs. Arm Stiffness', fontweight='bold', fontsize=11)
ax2.set_xlabel('Arm Stiffness K $\mathrm{N/m}$', fontsize=11)
ax2.set_xlim(k_line.min(), k_line.max())
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='upper right').set_zorder(7)

global_max = max(max(tau_vary_m_nom), max(tau_vary_k_nom))
ax1.set_ylim(0, global_max * 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11b_cross_sections.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11b_cross_sections.pdf'), format='pdf', bbox_inches='tight')
plt.close()

import matplotlib.colors as mcolors

# ── Fig 11c: Safety Margin Gap Plot ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

Z_gap = Z_tau - WCL_TARGET

vmin_val = min(-0.1, Z_gap.min()) 
vmax_val = max(0.1, Z_gap.max())
norm = mcolors.TwoSlopeNorm(vmin=vmin_val, vcenter=0, vmax=vmax_val)

cmap_diverging = sns.diverging_palette(15, 150, as_cmap=True)

cp = ax.contourf(X_M, Y_K, Z_gap, levels=40, cmap=cmap_diverging, norm=norm, alpha=0.9)
cbar = fig.colorbar(cp, ax=ax, pad=0.02)
# FIX 5: Simplified colorbar label
cbar.set_label('Safety Clearance Gap (ms)', fontsize=11, weight='bold')

# FIX 1: Heavy Failure Boundary (Unmissable)
ax.contour(X_M, Y_K, Z_gap, levels=[0], colors='black', linewidths=3.0, linestyles='-', zorder=4)
ax.contourf(X_M, Y_K, Z_gap, levels=[-np.inf, 0], colors=['none'], hatches=['////'], alpha=0.8, zorder=3)

ax.plot(M, K, marker='*', color='white', markersize=14, markeredgecolor='black', markeredgewidth=1.0, 
        label='Nominal Clearance', zorder=6)
ax.plot(min_tau_m, min_tau_k, marker='v', color='white', markersize=12, markeredgecolor='black', markeredgewidth=1.5, 
        label='Worst-Case Clearance', zorder=6)

ax.text(0.03, 0.04, r'Stable if Gap > 0',
        transform=ax.transAxes, fontsize=11, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round,pad=0.3'), zorder=7)

ax.set_title('Absolute Safety Clearance Gap (Zero = Instability)', fontweight='bold', fontsize=12)
ax.set_xlabel('Arm Mass M (kg)', fontsize=11)
ax.set_ylabel('Arm Stiffness K $\mathrm{N/m}$', fontsize=11)

ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black').set_zorder(7)
ax.grid(True, alpha=0.2, color='black')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11c_gap_plot.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_11c_gap_plot.pdf'), format='pdf', bbox_inches='tight')
plt.close()

# ── Fig 12: Noise sensitivity ─────────────────────────────────────────────
fig, (ax_main, ax_inset) = plt.subplots(1, 2, figsize=(13, 5))
t_arr = clean_r['t']
mask  = t_arr <= 2.0

ax_main.plot(t_arr[mask], clean_r['x'][mask],
             label=f'Clean baseline (IAE = {clean_IAE:.3f})',
             color='gray', lw=2.5, alpha=0.6, zorder=5)

for sigma, color in zip(NOISE_LEVELS, COLORS_NOISE):
    r_n  = noisy_results[sigma]
    pct  = int(sigma * 100)
    lbl  = fr'$\sigma={pct}\%$  IAE={r_n["mean_IAE"]:.3f}±{r_n["std_IAE"]:.3f} (mean-rep, N={NOISE_N_SEEDS})'
    if not r_n['bounded']:
        lbl += '  ⚠'
    ax_main.plot(t_arr[mask], r_n['x_repr'][mask],
                 color=color, lw=1.5, alpha=0.85, label=lbl)

ax_main.axhline(1.0,  color='gray', lw=1.0, ls='--', label='Reference', zorder=1)
ax_main.axhline(1.0 + SIM_BAND_TOLERANCE, color='gray', lw=0.5, ls=':', alpha=0.4)
ax_main.axhline(1.0 - SIM_BAND_TOLERANCE, color='gray', lw=0.5, ls=':', alpha=0.4)

worst_bounded = noisy_results[NOISE_LEVELS[-1]]['bounded']
ax_main.set_title(fr'Noise Sensitivity: Step Response at $WCL = {WCL_TARGET}\ \mathrm{{ms}}$'
                  + ('' if worst_bounded else '  [σ=10% oscillates, bounded]'),
                  fontweight='bold', fontsize=10)
ax_main.set_xlabel('Time (s)')
ax_main.set_ylabel(r'Position $x(t)$')
ax_main.legend(fontsize=6.5, loc='lower right')
ax_main.grid(True, alpha=0.25)

# Bar chart: mean IAE ± std
mean_IAEs  = [noisy_results[s]['mean_IAE'] for s in NOISE_LEVELS]
std_IAEs   = [noisy_results[s]['std_IAE']  for s in NOISE_LEVELS]
sigma_pcts = [int(s * 100) for s in NOISE_LEVELS]

max_y_val = max(m + s for m, s in zip(mean_IAEs, std_IAEs))
ax_inset.set_ylim(0, max_y_val * 1.20)

bar_x = np.arange(len(NOISE_LEVELS))
bars  = ax_inset.bar(bar_x, mean_IAEs, width=0.5,
                     color=COLORS_NOISE, alpha=0.85,
                     edgecolor='black', linewidth=0.8)
ax_inset.errorbar(bar_x, mean_IAEs, yerr=std_IAEs,
                  fmt='none', color='black', capsize=5, linewidth=1.5)
ax_inset.axhline(clean_IAE, color='gray', lw=1.5, ls='--', alpha=0.7,
                 label=f'Clean baseline IAE = {clean_IAE:.3f}')

for bar, mean, std, sigma in zip(bars, mean_IAEs, std_IAEs, NOISE_LEVELS):
    deg_pct    = (mean - clean_IAE) / clean_IAE * 100
    is_bounded = noisy_results[sigma]['bounded']
    lbl_text   = f'+{deg_pct:.1f}%' if deg_pct >= 0 else f'{deg_pct:.1f}%'
    if not is_bounded:
        lbl_text += '\n⚠'
    ax_inset.text(bar.get_x() + bar.get_width() / 2,
                  mean + std + max_y_val * 0.01,
                  lbl_text, ha='center', va='bottom', fontsize=9,
                  color='darkred' if not is_bounded else 'black')

ax_inset.set_xticks(bar_x)
ax_inset.set_xticklabels([fr'$\sigma={p}\%$' for p in sigma_pcts])
ax_inset.set_ylabel(f'Mean IAE  (N = {NOISE_N_SEEDS} seeds)')
ax_inset.set_title(f'IAE Degradation vs. Noise Level\n(error bars = ±1 std, N={NOISE_N_SEEDS} seeds)',
                   fontweight='bold')
ax_inset.legend(fontsize=8)
ax_inset.grid(True, alpha=0.25, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_12_noisy_response.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_12_noisy_response.pdf'), format='pdf', bbox_inches='tight')
plt.close()
print("✓ fig_12_noisy_response")

# ── 11. FIGURE 13 — DUAL-AXIS LOG PLOT (fig_9) ─────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

l1, = ax1.semilogy(tau_arr, [max(v, 0.01) for v in OS_arr], 'o-',
                   color='#1565C0', lw=2, markersize=8,
                   markerfacecolor='white', markeredgewidth=2, label='Overshoot (%)')
l2, = ax2.semilogy(tau_arr, ISE_arr, 's--',
                   color='#B71C1C', lw=2, markersize=8,
                   markerfacecolor='white', markeredgewidth=2, label='ISE')

ax1.axvspan(tau_max_ms, 55, alpha=0.08, color='red')
ax1.axvline(tau_max_ms, color='red',    lw=2.0, ls='-',
            label=fr'$\tau_{{\max}} = {tau_max_ms:.2f}\ \mathrm{{ms}}$')


merged_wcl = LATENCY_DATA['LoRA-CLIP (merged)']['p99'] + T_JITTER + T_BUFFER
unmerged_wcl = LATENCY_DATA['LoRA-CLIP (unmerged)']['p99'] + T_JITTER + T_BUFFER

ax1.axvline(merged_wcl, color='purple', lw=1.5, ls='--',
            label=f'LoRA-CLIP merged WCL ({merged_wcl:.2f} ms)')
ax1.axvline(unmerged_wcl, color='green',  lw=1.5, ls='--',
            label=f'LoRA-CLIP unmerged WCL ({unmerged_wcl:.2f} ms)')

ax1.set_xlabel(r'Perception Delay $\tau$ $\mathrm{ms}$', fontsize=11)
ax1.set_ylabel('Overshoot (%)', color='#1565C0', fontsize=11)
ax2.set_ylabel('ISE',           color='#B71C1C', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#1565C0')
ax2.tick_params(axis='y', labelcolor='#B71C1C')
ax1.set_title('Dual-Axis: Overshoot & ISE vs Perception Delay (log scale)', fontsize=11)
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(True, alpha=0.25)
ax1.set_xlim(7, 53)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'fig_9_dual_axis')
plt.savefig(f'{fig3_path}.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{fig3_path}.pdf', format='pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved {fig3_path}.png/.pdf")


# ── 11. SAVE JSON ─────────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, 'simulation_results.json')
with open(json_path, 'w') as f:
    json.dump(json_export_data, f, indent=4)
print(f"✓ {json_path}")

# ── 12. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("SUMMARY")
print("=" * 64)
print(f"  NumPy:             {np.__version__}  |  trapz fn: {_trapz.__name__}")
print(f"  tau_max (nominal)  = {tau_max_ms:.4f} ms")
print(f"  tau_max (grid min) = {min_tau_found:.2f} ms  "
      f"at M={min_tau_m:.2f} kg, K={min_tau_k:.0f} N/m")
print(f"  WCL target         = {WCL_TARGET} ms")
print(f"  Safety margin      = {safety_margin_ms:.2f} ms  "
      f"({'disclose in paper' if safety_margin_ms < 5.0 else 'adequate'})")
print(f"\n  Clean SiL metrics:")
for tau_ms in SWEEP_DELAY_VALUES_MS:
    r  = results[tau_ms]
    ts = f"{r['Ts']:.3f}" if r['Ts'] is not None else "N/A"
    print(f"    τ={tau_ms:>3}ms  IAE={r['IAE']:.4f}  OS={r['OS']:.2f}%  "
          f"ISE={r['ISE']:.4f}  Ts={ts}")
print(f"\n  Noise sweep (WCL={WCL_TARGET}ms, N={NOISE_N_SEEDS} seeds):")
for sigma in NOISE_LEVELS:
    r_n = noisy_results[sigma]
    eff = sigma * np.sqrt(2) * 100
    print(f"    σ={int(sigma*100):>3}% (eff {eff:.1f}%)  "
          f"mean={r_n['mean_IAE']:.4f}  std={r_n['std_IAE']:.4f}  "
          f"deg={r_n['deg_pct']:.1f}%  {'BOUNDED' if r_n['bounded'] else 'DIVERGENT'}")
print(f"\n  All outputs saved to: {OUTPUT_DIR}")