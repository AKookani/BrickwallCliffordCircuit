"""
Measurement-Induced Phase Transition (MIPT) — Brickwork Clifford Circuit
=========================================================================

Mirrors the Julia code (JuliaBrickCliff.jl) on a small system so you can
see every step concretely.

System layout  (L=4, so 2*L = 8 system qubits + 1 ancilla = 9 qubits total)
─────────────────────────────────────────────────────────────────────────────
  q0  q1  q2  q3 | q4  q5  q6  q7   q8(ancilla)
  ←──────────── system ────────────→  ↑ Bell-paired with middle qubit q3

Protocol
────────
  1. Bell pair            : entangle ancilla q8 with middle qubit q[L-1]=q3.
  2. Optional scrambling  : 4*L layers of random 2-qubit Clifford gates,
                            no measurements, to build volume-law entanglement.
  3. Monitored evolution  : T steps of
                              a) measure each system qubit in Z with prob p
                              b) brickwork layer of random 2-qubit Cliffords
  4. Purification check   : at each step test whether X/Y/Z on ancilla is
                            now deterministic → record purifyTime.
  5. Entanglement entropy : S(A) for the left half A=[0..L-1] using the
                            stabilizer rank formula  S = rank(proj_A) - |A|/2.
"""

import stim
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
#  Parameters
# ──────────────────────────────────────────────────────────────
L          = 4          # half-system size; total system qubits = 2*L = 8
T          = 24         # monitored time steps
p          = 0.10       # measurement rate
SCRAMBLE   = True       # pre-scramble before monitoring
N_SAMPLES  = 300        # trajectories for averaging
SEED       = 42
rng        = np.random.default_rng(SEED)

N_SYS   = 2 * L          # 8 system qubits
ANCILLA = N_SYS           # qubit index 8
N_TOT   = N_SYS + 1      # 9 total

print(f"\n{'='*60}")
print(f"  MIPT Brickwork Clifford  |  L={L}  N_sys={N_SYS}  p={p}")
print(f"  T={T}  scramble={SCRAMBLE}  samples={N_SAMPLES}")
print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────
#  Entanglement entropy via stabilizer rank  (GF2 row reduction)
# ──────────────────────────────────────────────────────────────
def gf2_rank(mat: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2) by Gaussian elimination."""
    M = mat.astype(np.uint8).copy()
    pivot_row = 0
    for col in range(M.shape[1]):
        # find pivot in this column at or below current pivot_row
        idx = np.nonzero(M[pivot_row:, col])[0]
        if idx.size == 0:
            continue
        r = pivot_row + idx[0]
        M[[pivot_row, r]] = M[[r, pivot_row]]
        # eliminate all other 1s in this column
        others = np.nonzero(M[:, col])[0]
        for o in others:
            if o != pivot_row:
                M[o] ^= M[pivot_row]
        pivot_row += 1
    return pivot_row


def stabilizer_entropy(sim: stim.TableauSimulator,
                       subsystem: list) -> float:
    """
    S(A) = rank( proj_A(stabilizer matrix) ) - |A|/2

    Build the n×2n binary stabilizer matrix from canonical_stabilizers(),
    project onto columns for qubits in `subsystem` (both X and Z columns),
    then take the GF(2) rank.  Exactly the Julia 'Entropy → clippingGauge'.
    """
    stabs = sim.canonical_stabilizers()
    n = len(stabs)
    # rows of the stabilizer matrix: [X_0..X_{n-1} | Z_0..Z_{n-1}]
    mat = np.zeros((n, 2 * n), dtype=np.uint8)
    for i, ps in enumerate(stabs):
        xs, zs = ps.to_numpy()
        mat[i, :n]  = xs.astype(np.uint8)
        mat[i, n:]  = zs.astype(np.uint8)

    # keep only columns belonging to subsystem A
    A_cols = list(subsystem) + [q + n for q in subsystem]
    proj   = mat[:, A_cols]
    return gf2_rank(proj) - len(subsystem) / 2.0


# ──────────────────────────────────────────────────────────────
#  Brickwork layer of random 2-qubit Cliffords
# ──────────────────────────────────────────────────────────────
def brickwork_layer(sim: stim.TableauSimulator,
                    n_sys: int, even: bool) -> None:
    """
    even=True  → pairs (0,1), (2,3), (4,5), ...
    even=False → pairs (1,2), (3,4), ...  + periodic wrap (n_sys-1, 0)
    """
    if even:
        pairs = [(2*i, 2*i+1) for i in range(n_sys // 2)]
    else:
        pairs = [(2*i+1, 2*i+2) for i in range((n_sys - 1) // 2)]
        pairs.append((n_sys - 1, 0))   # periodic boundary

    for a, b in pairs:
        gate = stim.Tableau.random(2)   # uniform over all 720 two-qubit Cliffords
        sim.do_tableau(gate, [a, b])


# ──────────────────────────────────────────────────────────────
#  Ancilla purification probe
# ──────────────────────────────────────────────────────────────
def ancilla_is_purified(sim: stim.TableauSimulator, ancilla: int,
                        n_tot: int) -> bool:
    """
    Returns True when at least one of X, Y, Z on the ancilla is deterministic.
    peek_observable_expectation returns ±1 if deterministic, 0 if random.
    The PauliString must cover all n_tot qubits (padded with I).
    """
    pad = 'I' * ancilla
    for pauli in ('X', 'Y', 'Z'):
        op = stim.PauliString(pad + pauli + 'I' * (n_tot - ancilla - 1))
        if sim.peek_observable_expectation(op) != 0:
            return True
    return False


# ──────────────────────────────────────────────────────────────
#  Single trajectory
# ──────────────────────────────────────────────────────────────
def run_trajectory(L: int, p: float, T: int, scramble: bool,
                   verbose: bool = False) -> dict:
    n_sys   = 2 * L
    ancilla = n_sys
    n_tot   = n_sys + 1
    left    = list(range(L))   # subsystem A = left half of system

    sim = stim.TableauSimulator()
    # stim starts in |0⟩^⊗n  (Z-stabilised product state) automatically.

    # ── Bell pair: ancilla ↔ middle qubit ──────────────────────────────────
    # Middle qubit = L-1.  We create |Φ+⟩ = (|00⟩+|11⟩)/√2 via H + CNOT.
    middle = L - 1
    sim.do(stim.CircuitInstruction("H",    [ancilla]))
    sim.do(stim.CircuitInstruction("CNOT", [ancilla, middle]))

    if verbose:
        S0 = stabilizer_entropy(sim, left)
        print(f"  [init] S(left half) = {S0:.3f}  (Bell pair with qubit {middle})")

    # ── Optional scrambling ────────────────────────────────────────────────
    if scramble:
        for t in range(4 * L):
            brickwork_layer(sim, n_sys, even=(t % 2 == 0))
        if verbose:
            Ss = stabilizer_entropy(sim, left)
            print(f"  [after scramble] S(left half) = {Ss:.3f}")

    # ── Monitored evolution ────────────────────────────────────────────────
    entropy_t   = []
    purify_time = None
    meas_record = np.full((T, n_sys), np.nan)   # NaN = no measurement

    for t in range(T):
        # (a) Random Z-measurements on each system qubit
        for i in range(n_sys):
            if rng.random() < p:
                outcome = sim.measure(i)             # collapses qubit i in Z
                meas_record[t, i] = +1 if outcome else -1

        # (b) Brickwork Clifford layer (even/odd alternating)
        brickwork_layer(sim, n_sys, even=(t % 2 == 0))

        # Track entropy of left half
        S = stabilizer_entropy(sim, left)
        entropy_t.append(S)

        # Check ancilla purification
        if purify_time is None and ancilla_is_purified(sim, ancilla, n_tot):
            purify_time = t
            if verbose:
                print(f"  [t={t:2d}] Ancilla purified!  S = {S:.3f}")

    return {
        "entropy_vs_time":    entropy_t,
        "purify_time":        purify_time,
        "measurement_record": meas_record,
    }


# ──────────────────────────────────────────────────────────────
#  Average over many samples
# ──────────────────────────────────────────────────────────────
def run_experiment(L, p, T, scramble, n_samples):
    all_entropy  = np.zeros((n_samples, T))
    purify_times = []
    for s in range(n_samples):
        if s % 100 == 0:
            print(f"    sample {s}/{n_samples}")
        res = run_trajectory(L, p, T, scramble, verbose=(s == 0 and p == 0.05))
        all_entropy[s] = res["entropy_vs_time"]
        pt = res["purify_time"]
        purify_times.append(pt if pt is not None else T)
    return all_entropy.mean(axis=0), np.mean(purify_times), all_entropy


# ──────────────────────────────────────────────────────────────
#  Scan p values
# ──────────────────────────────────────────────────────────────
p_values = [0.05, 0.10, 0.16, 0.20, 0.30, 0.40]
results  = {}

for pv in p_values:
    print(f"\n── p = {pv} ──")
    mean_S, mean_pt, all_S = run_experiment(L, pv, T, SCRAMBLE, N_SAMPLES)
    results[pv] = {"mean_S": mean_S, "mean_pt": mean_pt}
    print(f"   mean purify time = {mean_pt:.1f}    final S = {mean_S[-1]:.3f}")


# ──────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"MIPT Brickwork Clifford  —  L={L}, N_sys={N_SYS}, T={T}, "
    f"{N_SAMPLES} samples",
    fontsize=13
)

# Plot 1: S(t) for all p
ax = axes[0]
for pv in p_values:
    ax.plot(results[pv]["mean_S"], label=f"p={pv:.2f}")
ax.set_xlabel("Time step t")
ax.set_ylabel(r"$\langle S(A) \rangle$")
ax.set_title("Mean entanglement entropy (left half)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: final S vs p  — the clearest signature of the phase transition
ax = axes[1]
final_S = [results[pv]["mean_S"][-1] for pv in p_values]
ax.plot(p_values, final_S, 'o-', color='steelblue', linewidth=2, markersize=8)
ax.axvline(x=0.16, color='red', linestyle='--', alpha=0.7, label=r'$p_c \approx 0.16$')
ax.set_xlabel("Measurement rate p")
ax.set_ylabel(r"$\langle S(A) \rangle$ at $t=T$")
ax.set_title("Final entropy vs measurement rate")
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: mean purification time vs p
ax = axes[2]
mean_pts = [results[pv]["mean_pt"] for pv in p_values]
ax.plot(p_values, mean_pts, 's-', color='darkorange', linewidth=2, markersize=8)
ax.axvline(x=0.16, color='red', linestyle='--', alpha=0.7, label=r'$p_c \approx 0.16$')
ax.set_xlabel("Measurement rate p")
ax.set_ylabel("Mean purification time")
ax.set_title("Ancilla purification time vs p")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/mipt_results.png", dpi=150, bbox_inches='tight')
print("\nPlot saved.")


# ──────────────────────────────────────────────────────────────
#  Print one trajectory's measurement record (demo)
# ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Demo: single trajectory  (p=0.10)")
print(f"  Columns = system qubits 0..{N_SYS-1}")
print(f"  +1 = measured ↑   -1 = measured ↓    . = not measured")
print(f"{'='*60}")

res_demo = run_trajectory(L=L, p=0.10, T=T, scramble=SCRAMBLE, verbose=True)
rec = res_demo["measurement_record"]
for t in range(min(10, T)):
    row = "".join(
        f"  ." if np.isnan(rec[t, i]) else f" {int(rec[t,i]):+d}"
        for i in range(N_SYS)
    )
    S_str = f"  S={res_demo['entropy_vs_time'][t]:.2f}"
    print(f"  t={t:2d}  |{row}  |{S_str}")

pt = res_demo["purify_time"]
print(f"\n  Ancilla purified at t = {pt if pt is not None else 'never'}")
print(f"  Final S(left half)  = {res_demo['entropy_vs_time'][-1]:.4f}\n")
