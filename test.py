import numpy as np
import torch
import os
import time
import datetime

# ---------------- Sp(4,2) precomputation (720 elements, cached) ----------------

def build_sp42(cache_path="sp42_table.npy"):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    idx = np.arange(1 << 16, dtype=np.uint32)
    mats = ((idx[:, None] >> np.arange(16)) & 1).astype(np.int64).reshape(-1, 4, 4)
    J = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    lhs = np.einsum('bji,jk,bkl->bil', mats, J, mats) % 2   # M^T J M
    table = mats[(lhs == J).all(axis=(1, 2))].astype(np.uint8)
    assert len(table) == 720
    np.save(cache_path, table)
    return table


def build_lut(table, device):
    """(720, 16) lookup: gate index, packed 4-bit input -> packed 4-bit output."""
    s = np.arange(16)
    bits = (s[:, None] >> np.arange(4)) & 1                                   # (16,4)
    new = (table[:, None].astype(int) * bits[None, :, None, :]).sum(-1) % 2   # (720,16,4)
    packed = (new * np.array([1, 2, 4, 8])).sum(-1)
    return torch.tensor(packed, dtype=torch.uint8, device=device)

def build_sign_lut(table, device, cache_path="sp42_signs.npy"):
    """(720,16): sign bit picked up by Hermitian Pauli s under a fixed Clifford
    representative of each symplectic matrix (s uses the same 4-bit packing as the LUT)."""
    if os.path.exists(cache_path):
        return torch.tensor(np.load(cache_path), dtype=torch.uint8, device=device)
    Xm = np.array([[0, 1], [1, 0]], dtype=complex)
    Zm = np.diag([1, -1]).astype(complex)

    def herm(x, z):                                   # Hermitian single-qubit Pauli (Y stays Y)
        m = np.eye(2, dtype=complex)
        if x: m = m @ Xm
        if z: m = m @ Zm
        return (1j ** (int(x) * int(z))) * m

    vecs = [np.array([(s >> i) & 1 for i in range(4)]) for s in range(16)]
    P = [np.kron(herm(v[0], v[1]), herm(v[2], v[3])) for v in vecs]
    pack = lambda v: int(sum(int(v[r]) << r for r in range(4)))
    out = np.zeros((len(table), 16), dtype=np.uint8)
    for gi, S in enumerate(table.astype(int)):
        img = [P[pack(S[:, i])] for i in range(4)]    # image of each generator, sign +
        for s in range(16):
            W = np.eye(4, dtype=complex); V = np.eye(4, dtype=complex)
            for i in range(4):
                if (s >> i) & 1:
                    W = W @ P[1 << i]
                    V = V @ img[i]
            t = pack(S @ vecs[s] % 2)
            c = np.trace(P[s].conj().T @ W) / 4       # W = c * P_s
            d = np.trace(P[t].conj().T @ V) / 4       # V = d * P_{S s}
            out[gi, s] = 0 if (d / c).real > 0 else 1
    np.save(cache_path, out)
    return torch.tensor(out, dtype=torch.uint8, device=device)


def make_pairs(L, periodic, device):
    assert L % 2 == 0, "use even L"
    out = []
    for off in (0, 1):
        p = [(j, j + 1) for j in range(off, L - 1, 2)]
        if periodic and off == 1:
            p.append((L - 1, 0))
        a = torch.tensor([x for x, _ in p], device=device)
        b = torch.tensor([y for _, y in p], device=device)
        out.append((a, b))
    return out


# ---------------- Layers ----------------

def brickwork_layer(X, Z, R, t, pairs, lut, sign_lut, gen=None):
    pa, pb = pairs[t % 2]
    B, n2, _ = X.shape
    P = pa.numel()
    s = (X[:, :, pa] | (Z[:, :, pa] << 1) | (X[:, :, pb] << 2) | (Z[:, :, pb] << 3)).long()
    g = torch.randint(0, lut.shape[0], (B, 1, P), device=X.device, generator=gen)
    q = torch.randint(0, 16, (B, 1, P), device=X.device, generator=gen)   # random Pauli frame
    out = lut[g, s]
    # sign: representative's sign ^ anticommutation with the random Pauli q
    flip = (((s & 1) & ((q >> 1) & 1)) ^ (((s >> 1) & 1) & (q & 1)) ^
            (((s >> 2) & 1) & ((q >> 3) & 1)) ^ (((s >> 3) & 1) & ((q >> 2) & 1)))
    bits = sign_lut[g, s].long() ^ flip
    R ^= (bits.sum(-1) & 1).to(torch.uint8)
    X[:, :, pa] = out & 1
    Z[:, :, pa] = (out >> 1) & 1
    X[:, :, pb] = (out >> 2) & 1
    Z[:, :, pb] = (out >> 3) & 1


def _g(x1, z1, x2, z2):
    """Aaronson-Gottesman phase exponent (power of i) for P1*P2, int8 inputs."""
    return torch.where((x1 & z1).bool(), z2 - x2,
           torch.where(x1.bool(), z2 * (2 * x2 - 1),
           torch.where(z1.bool(), x2 * (1 - 2 * z2), torch.zeros_like(x1))))


def measure_layer(X, Z, R, p, t, rec, gen=None):
    B, N2, N = X.shape                                 # N2 = 2N rows (destab, stab)
    dev = X.device
    do = torch.rand((B, N), device=dev, generator=gen) < p
    counts = do.sum(1)
    K = int(counts.max())
    if K == 0:
        return
    qs = torch.argsort(do.to(torch.uint8), dim=1, descending=True, stable=True)[:, :K]
    b = torch.arange(B, device=dev)
    for k in range(K):
        q = qs[:, k]
        active = k < counts
        anti = X[b, :, q].bool()                       # (B,2N) rows anticommuting with Z_q
        anti_s = anti[:, N:]
        has_anti = anti_s.any(1)
        is_rand = has_anti & active
        det = (~has_anti) & active
        outcome = torch.zeros(B, dtype=torch.uint8, device=dev)

        if is_rand.any():                              # ---- random outcome
            piv = N + anti_s.to(torch.uint8).argmax(1)
            pX, pZ, pR = X[b, piv], Z[b, piv], R[b, piv]
            upd = anti & is_rand[:, None]
            upd[b, piv] = False
            gsum = _g(X.to(torch.int8), Z.to(torch.int8),
                      pX.to(torch.int8)[:, None], pZ.to(torch.int8)[:, None]).sum(-1, dtype=torch.int32)
            k4 = gsum + 2 * R.int() + 2 * pR.int()[:, None]
            R.copy_(torch.where(upd, (torch.remainder(k4, 4) // 2).to(torch.uint8), R))
            u = upd.to(torch.uint8).unsqueeze(-1)
            X ^= u * pX[:, None, :]
            Z ^= u * pZ[:, None, :]
            d = piv - N                                # destabilizer partner := old pivot
            X[b, d] = torch.where(is_rand[:, None], pX, X[b, d])
            Z[b, d] = torch.where(is_rand[:, None], pZ, Z[b, d])
            m = torch.randint(0, 2, (B,), device=dev, generator=gen).to(torch.uint8)
            oh = torch.zeros_like(pZ); oh[b, q] = 1
            X[b, piv] = torch.where(is_rand[:, None], torch.zeros_like(pX), pX)
            Z[b, piv] = torch.where(is_rand[:, None], oh, pZ)
            R[b, piv] = torch.where(is_rand, m, pR)
            outcome = torch.where(is_rand, m, outcome)

        if det.any():                                  # ---- deterministic outcome
            idx = det.nonzero().squeeze(1)
            sel = anti[idx, :N].to(torch.int8)         # destabilizers hitting Z_q -> stabilizers to multiply
            Xs = X[idx, N:].to(torch.int8) * sel[:, :, None]
            Zs = Z[idx, N:].to(torch.int8) * sel[:, :, None]
            px = ((torch.cumsum(Xs, 1, dtype=torch.int16) - Xs) % 2).to(torch.int8)   # running product
            pz = ((torch.cumsum(Zs, 1, dtype=torch.int16) - Zs) % 2).to(torch.int8)
            gsum = _g(px, pz, Xs, Zs).sum((1, 2), dtype=torch.int32)
            k4 = gsum + 2 * (R[idx, N:].int() * sel).sum(1)
            outcome[idx] = (torch.remainder(k4, 4) // 2).to(torch.uint8)

        if rec is not None:
            a = active.nonzero().squeeze(1)
            rec.append((t, a.int(), q[a].short(), outcome[a], is_rand[a].to(torch.uint8)))


# ---------------- Entropy ----------------

def gf2_rank_batched(M, device):
    batch, rows, cols = M.shape
    word = 62                                          # fits in int64
    num_words = (cols + word - 1) // word
    pad = num_words * word - cols
    if pad:
        M = torch.nn.functional.pad(M, (0, pad))
    M_r = M.view(batch, rows, num_words, word).to(torch.int64)
    weights = 2 ** torch.arange(word, device=device, dtype=torch.int64)
    packed = (M_r * weights).sum(dim=-1)

    rank_ptr = torch.zeros(batch, dtype=torch.int64, device=device)
    b_idx = torch.arange(batch, device=device)
    row_ar = torch.arange(rows, device=device)
    idx_grid = row_ar.unsqueeze(0).expand(batch, rows)

    for c in range(cols):
        w, bit = c // word, c % word
        col_bits = (packed[:, :, w] >> bit) & 1
        candidate = (row_ar.unsqueeze(0) >= rank_ptr.unsqueeze(1)) & (col_bits == 1)
        masked_idx = torch.where(candidate, idx_grid, torch.full_like(idx_grid, rows))
        pivot_idx = masked_idx.min(dim=1).values
        has_pivot = pivot_idx < rows
        safe_rank = torch.clamp(rank_ptr, max=rows - 1)
        safe_piv = torch.clamp(torch.where(has_pivot, pivot_idx, rank_ptr), max=rows - 1)

        row_rank = packed[b_idx, safe_rank, :].clone()
        row_piv = packed[b_idx, safe_piv, :].clone()
        packed[b_idx, safe_rank, :] = torch.where(has_pivot.unsqueeze(-1), row_piv, row_rank)
        packed[b_idx, safe_piv, :] = torch.where(has_pivot.unsqueeze(-1), row_rank, row_piv)

        col_after = (packed[:, :, w] >> bit) & 1
        elim = col_after.clone()
        elim[b_idx, safe_rank] = 0
        elim = elim * has_pivot.unsqueeze(1).to(torch.int64)
        packed ^= elim.unsqueeze(-1) * packed[b_idx, safe_rank, :].unsqueeze(1)
        rank_ptr = rank_ptr + has_pivot.to(torch.int64)
    return rank_ptr


def half_chain_entropy_batched(X, Z, region, device):
    M_A = torch.cat([X[:, :, region], Z[:, :, region]], dim=-1)
    return (gf2_rank_batched(M_A, device) - region.numel()).float()


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------- Driver ----------------

def run_averaged_gpu(size, T, p_measure, n_trials, lut, sign_lut, device,
                     periodic=True, entropy_every=2, seed=None, record=True):
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    n = size
    pairs = make_pairs(size, periodic, device)
    region_a = torch.arange(size // 2, device=device)

    # rows 0..n-1 destabilizers (X_i), rows n..2n-1 stabilizers (Z_i)
    X = torch.zeros((n_trials, 2 * n, n), dtype=torch.uint8, device=device)
    Z = torch.zeros((n_trials, 2 * n, n), dtype=torch.uint8, device=device)
    R = torch.zeros((n_trials, 2 * n), dtype=torch.uint8, device=device)
    ar = torch.arange(n)
    X[:, ar, ar] = 1
    Z[:, n + ar, ar] = 1

    ent = lambda: half_chain_entropy_batched(X[:, n:], Z[:, n:], region_a, device)
    rec = [] if record else None
    ts = [0]
    hist = [ent()]
    for t in range(1, T):
        brickwork_layer(X, Z, R, t, pairs, lut, sign_lut, gen)
        measure_layer(X, Z, R, p_measure, t, rec, gen)
        if t % entropy_every == 0:
            ts.append(t)
            hist.append(ent())

    H = torch.stack(hist, dim=1)
    mean_S = H.mean(0).cpu().numpy()
    sem_S = (H.std(0) / n_trials ** 0.5).cpu().numpy()

    data = None
    if record:
        if rec:
            data = dict(
                trial=torch.cat([r[1] for r in rec]).cpu().numpy(),
                t=torch.cat([torch.full_like(r[1], r[0], dtype=torch.int16) for r in rec]).cpu().numpy(),
                qubit=torch.cat([r[2] for r in rec]).cpu().numpy(),
                outcome=torch.cat([r[3] for r in rec]).cpu().numpy(),
                was_random=torch.cat([r[4] for r in rec]).cpu().numpy())
        else:
            data = {k: np.zeros(0, dtype=np.int8) for k in
                    ("trial", "t", "qubit", "outcome", "was_random")}
    return np.array(ts), mean_S, sem_S, data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    table = build_sp42()
    lut = build_lut(table, device)
    sign_lut = build_sign_lut(table, device)

    size = 128
    T = 2 * size
    n_trials = 1000
    p_values = np.round(np.arange(0.0, 0.2 + 1e-9, 0.02), 2)

    plt.figure(figsize=(8, 6))
    run_times = []
    n_total = len(p_values)
    overall_start = time.time()
    cmap = plt.cm.jet

    for i, p in enumerate(p_values):
        t0 = time.time()
        ts, mean_S, sem_S, data = run_averaged_gpu(size, T, float(p), n_trials, lut, sign_lut, device)
        np.savez_compressed(f"mipt_data_L{size}_p{p:.2f}.npz", p=p, L=size, T=T, **data)
        dt = time.time() - t0
        run_times.append(dt)
        remaining = sum(run_times) / len(run_times) * (n_total - i - 1)
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
        print(f"[{i+1}/{n_total}] p={p:.2f} done in {format_duration(dt)} | "
              f"remaining ~{format_duration(remaining)} (ETA {eta.strftime('%H:%M:%S')})")

        late = ts >= 2 * size
        print(f"    late-time mean S = {mean_S[late].mean():.2f}")

        plt.errorbar(ts, mean_S, yerr=sem_S, marker='o', markersize=3,
                     label=f"p={p:.2f}", color=cmap(i / (n_total - 1)))

    print(f"Total wall time: {format_duration(time.time() - overall_start)}")
    plt.xlabel("Time step")
    plt.ylabel("Half-chain entanglement entropy (bits)")
    plt.title(f"Half-chain entropy, {n_trials} trajectories, L={size}")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("mipt_result.png", dpi=150)
    plt.show()