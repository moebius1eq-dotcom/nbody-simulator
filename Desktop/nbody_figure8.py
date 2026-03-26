"""
N-Body Gravitational Simulator — Figure-Eight Three-Body Solution
Integrator : Velocity Verlet (symplectic, 2nd-order)
Reference   : Chenciner & Montgomery (2000), "A remarkable solution of the
              three body problem in the case of equal masses"
Units       : G = 1, m = 1 (natural / dimensionless)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# ── 1. Physical constants & bodies ──────────────────────────────────────────
G      = 1.0
MASSES = np.array([1.0, 1.0, 1.0])          # equal masses
N      = len(MASSES)

# ── 2. Figure-Eight Initial Conditions ──────────────────────────────────────
# Chenciner & Montgomery (2000) / Šuvakov & Dmitrašinović (2013)
#   r1 = –r2,  r3 = 0
#   v1 = v2 = –v3/2  (centre-of-mass momentum = 0)
_v3x, _v3y = 0.93240737, 0.86473146

R0 = np.array([
    [-0.97000436,  0.24308753],
    [ 0.97000436, -0.24308753],
    [ 0.0,         0.0       ],
], dtype=np.float64)

V0 = np.array([
    [-_v3x / 2, -_v3y / 2],
    [-_v3x / 2, -_v3y / 2],
    [ _v3x,      _v3y     ],
], dtype=np.float64)

# ── 3. Simulation parameters ─────────────────────────────────────────────────
T_PERIOD   = 6.3259          # one choreography period (natural units)
N_PERIODS  = 3               # simulate 3 full orbits
T_TOTAL    = N_PERIODS * T_PERIOD
DT         = 5e-5            # timestep — Verlet is 2nd-order, keep this tight
N_STEPS    = int(T_TOTAL / DT)
STORE_EVERY = 200            # subsample for animation frames

print(f"Figure-Eight N-Body Simulator")
print(f"{'─'*50}")
print(f"Integrator  : Velocity Verlet")
print(f"Bodies      : {N}  (m = 1 each, G = 1)")
print(f"Timestep    : {DT:.1e}")
print(f"Total steps : {N_STEPS:,}  over  t = {T_TOTAL:.4f}")
print(f"Periods     : {N_PERIODS}  (T ≈ {T_PERIOD})")
print()

# ── 4. Vectorised force / acceleration ───────────────────────────────────────
def accelerations(r: np.ndarray, masses: np.ndarray, G: float) -> np.ndarray:
    """
    Compute gravitational accelerations for all bodies in one vectorised pass.

    Parameters
    ----------
    r      : (N, 2) positions
    masses : (N,)  masses
    G      : gravitational constant

    Returns
    -------
    a      : (N, 2) accelerations  (F_i / m_i)
    """
    # displacement vectors: dr[i, j] = r[j] - r[i],  shape (N, N, 2)
    dr      = r[np.newaxis, :, :] - r[:, np.newaxis, :]
    # squared distances, shape (N, N)
    dist2   = np.sum(dr * dr, axis=-1)
    np.fill_diagonal(dist2, np.inf)          # avoid 1/0 on diagonal
    dist3   = dist2 ** 1.5
    # scalar factor G*m_j / |r_ij|^3 for each (i, j), shape (N, N)
    factor  = G * masses[np.newaxis, :] / dist3
    # sum contributions: a_i = Σ_j  factor[i,j] * dr[i,j]
    return np.einsum("ij,ijk->ik", factor, dr)


# ── 5. Total energy (K + U) ───────────────────────────────────────────────────
def total_energy(r: np.ndarray, v: np.ndarray,
                 masses: np.ndarray, G: float) -> float:
    """
    E = ½ Σ m_i |v_i|² − Σ_{i<j} G m_i m_j / |r_ij|
    """
    K = 0.5 * np.sum(masses[:, np.newaxis] * v * v)

    dr    = r[np.newaxis, :, :] - r[:, np.newaxis, :]
    dist  = np.sqrt(np.sum(dr * dr, axis=-1))
    np.fill_diagonal(dist, np.inf)
    # upper-triangle only to avoid double-counting
    i_idx, j_idx = np.triu_indices(N, k=1)
    U = -G * np.sum(masses[i_idx] * masses[j_idx] / dist[i_idx, j_idx])

    return K + U


# ── 6. Velocity-Verlet integration ───────────────────────────────────────────
n_stored = (N_STEPS - 1) // STORE_EVERY + 1
traj     = np.empty((n_stored, N, 2), dtype=np.float64)
E_trace  = np.empty(n_stored,          dtype=np.float64)

r = R0.copy()
v = V0.copy()
a = accelerations(r, MASSES, G)          # seed accelerations

E0 = total_energy(r, v, MASSES, G)
print(f"Initial energy  E₀ = {E0:.10f}")
print(f"Running integration …")

store_idx = 0
for step in range(N_STEPS):
    # — position update (uses current a) ——————————————————
    r = r + v * DT + 0.5 * a * DT * DT

    # — new acceleration at updated position ———————————————
    a_new = accelerations(r, MASSES, G)

    # — velocity update (average of old & new accel) ———————
    v = v + 0.5 * (a + a_new) * DT

    a = a_new

    if step % STORE_EVERY == 0:
        traj[store_idx]    = r
        E_trace[store_idx] = total_energy(r, v, MASSES, G)
        store_idx         += 1

n_stored = store_idx          # actual frames stored

E_final   = E_trace[n_stored - 1]
drift_abs = abs(E_final - E0)
drift_pct = drift_abs / abs(E0) * 100.0

print()
print(f"{'═'*50}")
print(f"  ENERGY CONSERVATION REPORT")
print(f"{'─'*50}")
print(f"  E₀  (initial) = {E0:.10f}")
print(f"  E_f (final)   = {E_final:.10f}")
print(f"  |ΔE|          = {drift_abs:.2e}")
print(f"  |ΔE/E₀| × 100 = {drift_pct:.6f} %")
print(f"{'═'*50}")
print()

# ── 7. Animation ─────────────────────────────────────────────────────────────
COLORS     = ["#FF6B6B", "#4ECDC4", "#FFE66D"]
BODY_NAMES = ["Body 1", "Body 2", "Body 3"]
TRAIL_FRAMES = 600           # fade-out trail length

fig = plt.figure(figsize=(15, 7), facecolor="#080c14")
fig.suptitle(
    "Figure-Eight Three-Body Solution  —  Velocity Verlet",
    color="white", fontsize=14, fontweight="bold", y=0.97
)

# left: orbit panel
ax_orb = fig.add_axes([0.03, 0.06, 0.55, 0.88], facecolor="#080c14")
ax_orb.set_aspect("equal")
ax_orb.set_xlim(-1.6, 1.6)
ax_orb.set_ylim(-1.1, 1.1)
ax_orb.tick_params(colors="#555")
for sp in ax_orb.spines.values():
    sp.set_color("#222")
ax_orb.set_xlabel("x", color="#888")
ax_orb.set_ylabel("y", color="#888")
ax_orb.set_title("Orbital Trajectories", color="#aaa", fontsize=11)

# right: energy panel
ax_e = fig.add_axes([0.64, 0.55, 0.33, 0.38], facecolor="#080c14")
ax_e.tick_params(colors="#555", labelsize=8)
for sp in ax_e.spines.values():
    sp.set_color("#222")
ax_e.set_xlabel("frame", color="#888", fontsize=8)
ax_e.set_ylabel("E  (K+U)", color="#888", fontsize=8)
ax_e.set_title("Total Energy", color="#aaa", fontsize=10)
ax_e.set_xlim(0, n_stored)
pad = 0.05 * abs(E_trace[:n_stored].max() - E_trace[:n_stored].min() + 1e-12)
ax_e.set_ylim(E_trace[:n_stored].min() - pad,
              E_trace[:n_stored].max() + pad)

# energy drift percentage panel
ax_d = fig.add_axes([0.64, 0.08, 0.33, 0.38], facecolor="#080c14")
ax_d.tick_params(colors="#555", labelsize=8)
for sp in ax_d.spines.values():
    sp.set_color("#222")
ax_d.set_xlabel("frame", color="#888", fontsize=8)
ax_d.set_ylabel("|ΔE/E₀| %", color="#888", fontsize=8)
ax_d.set_title("Relative Energy Drift", color="#aaa", fontsize=10)
ax_d.set_xlim(0, n_stored)
drift_trace = np.abs((E_trace[:n_stored] - E0) / E0) * 100.0
ax_d.set_ylim(0, max(drift_trace.max() * 1.3, 1e-8))

# — artists ——————————————————————————————————————————————————
trails = [ax_orb.plot([], [], "-", color=c, alpha=0.55,
                      linewidth=1.5, label=n)[0]
          for c, n in zip(COLORS, BODY_NAMES)]
dots   = [ax_orb.plot([], [], "o", color=c, markersize=11,
                      markeredgecolor="white", markeredgewidth=0.6)[0]
          for c in COLORS]

ax_orb.legend(loc="upper right", facecolor="#111", labelcolor="white",
              fontsize=9, framealpha=0.7)

time_txt  = ax_orb.text(0.02, 0.97, "", transform=ax_orb.transAxes,
                        color="white", fontsize=9, va="top",
                        fontfamily="monospace")
drift_txt = ax_orb.text(0.02, 0.90, "", transform=ax_orb.transAxes,
                        color="#FFE66D", fontsize=9, va="top",
                        fontfamily="monospace")

e_line,  = ax_e.plot([], [], "-",  color="#4ECDC4", linewidth=1.2)
e_dot,   = ax_e.plot([], [], "o",  color="#FF6B6B", markersize=5)
d_line,  = ax_d.plot([], [], "-",  color="#FFE66D", linewidth=1.2)
d_dot,   = ax_d.plot([], [], "o",  color="#FF6B6B", markersize=5)


def _init():
    for t in trails: t.set_data([], [])
    for d in dots:   d.set_data([], [])
    e_line.set_data([], [])
    e_dot.set_data( [], [])
    d_line.set_data([], [])
    d_dot.set_data( [], [])
    time_txt.set_text("")
    drift_txt.set_text("")
    return trails + dots + [e_line, e_dot, d_line, d_dot,
                             time_txt, drift_txt]


def _update(frame):
    start = max(0, frame - TRAIL_FRAMES)
    for i in range(N):
        trails[i].set_data(traj[start:frame+1, i, 0],
                           traj[start:frame+1, i, 1])
        dots[i].set_data([traj[frame, i, 0]], [traj[frame, i, 1]])

    frames = np.arange(frame + 1)
    e_line.set_data(frames, E_trace[:frame+1])
    e_dot.set_data( [frame], [E_trace[frame]])
    d_line.set_data(frames, drift_trace[:frame+1])
    d_dot.set_data( [frame], [drift_trace[frame]])

    t_sim  = frame * DT * STORE_EVERY
    dp     = drift_trace[frame]
    time_txt.set_text(f"t = {t_sim:6.3f}")
    drift_txt.set_text(f"|ΔE/E₀| = {dp:.4e} %")
    return trails + dots + [e_line, e_dot, d_line, d_dot,
                             time_txt, drift_txt]


print("Building animation …")
ani = animation.FuncAnimation(
    fig, _update, frames=n_stored,
    init_func=_init, interval=16, blit=True
)

output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "figure_eight_nbody.mp4")
writer = animation.FFMpegWriter(
    fps=60, bitrate=4000,
    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "slow", "-crf", "18"]
)
print(f"Saving → {output}")
ani.save(output, writer=writer, dpi=150)
plt.close(fig)

print(f"\nDone.  Video: {output}")
print(f"Final energy drift: {drift_pct:.6f} %")
