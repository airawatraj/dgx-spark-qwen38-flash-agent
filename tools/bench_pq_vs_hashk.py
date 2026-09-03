#!/usr/bin/env python3
"""Benchmark and compare PLE compression methods: HashK (R=4) vs Product Quantization (PQ).

Compares row reconstruction fidelity on held-out rows from the Qwen3.8-Flash-Next
PLE n-gram embedding table (OCP FP8 E4M3, dim=160).

Can read directly from a local safetensors shard (e.g. model-plefp8-00000.safetensors)
without third-party dependencies (numpy only, runs entirely on CPU in ~50s).

Based on comparative research and findings by @jucedik on the NVIDIA Developer Forums:
https://gist.github.com/Jucedik23/ca21eea17846d26ff3ce8090c54d6564
"""
import glob, json, os, sys, time
import numpy as np

DIM = 160
NUM_ROWS = 600_000  # ~96 MB of raw FP8 rows
SEED = 12345


def fp8_e4m3_lut():
    """256-entry lookup table: byte -> float32 for OCP E4M3 (bias 7)."""
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = (b >> 7) & 1
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            out[b] = np.nan  # NaN in E4M3
        elif e == 0:
            out[b] = (-1.0) ** s * (2.0 ** -6) * (m / 8.0)
        else:
            out[b] = (-1.0) ** s * (2.0 ** (e - 7)) * (1.0 + m / 8.0)
    return out


def find_default_shard():
    hf_cache = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    pattern = os.path.join(
        hf_cache,
        "hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/model-plefp8-00000.safetensors",
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_safetensors_slice(path, num_rows=NUM_ROWS, dim=DIM):
    print(f"Reading {num_rows:,} rows ({num_rows * dim / 1e6:.1f} MB) from {path}...")
    with open(path, "rb") as f:
        hdr_len = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        first_key = [k for k in hdr.keys() if k != "__metadata__"][0]
        data_start = 8 + hdr_len + hdr[first_key]["data_offsets"][0]
        f.seek(data_start)
        raw = np.fromfile(f, dtype=np.uint8, count=num_rows * dim)
    n = raw.size // dim
    lut = fp8_e4m3_lut()
    x = lut[raw[: n * dim].reshape(n, dim)]
    bad = ~np.isfinite(x).all(axis=1)
    if bad.any():
        x = x[~bad]
        print(f"  Dropped {bad.sum()} rows containing NaN")
    return x


def cosine(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    c = np.zeros(len(a), dtype=np.float64)
    c[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
    return c[ok]


def assign(x, c):
    """Assign points to nearest centroid using BLAS matmul."""
    cn = (c * c).sum(1)
    lab = np.empty(len(x), dtype=np.int32)
    step = 200_000
    for i in range(0, len(x), step):
        blk = x[i : i + step]
        lab[i : i + step] = (cn[None, :] - 2.0 * (blk @ c.T)).argmin(1)
    return lab


def kmeans(x, k, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        lab = assign(x, c)
        cnt = np.bincount(lab, minlength=k).astype(np.float32)
        new = np.zeros_like(c)
        for d in range(x.shape[1]):
            new[:, d] = np.bincount(lab, weights=x[:, d], minlength=k)
        live = cnt > 0
        new[live] /= cnt[live, None]
        new[~live] = c[~live]
        if np.allclose(new, c, atol=1e-6):
            break
        c = new
    return c


def hashk(train, test, R=4, seed=0):
    """Simulate HashK R=4 mean-pooling and ridge regression."""
    rng = np.random.default_rng(seed)

    def pool(x):
        perm = rng.permutation(len(x) // R * R)
        g = perm.reshape(-1, R)
        slots = x[g].mean(1)
        recon = np.repeat(slots, R, axis=0)
        truth = x[g.reshape(-1)]
        return recon, truth

    rec_tr, tru_tr = pool(train)
    lam = 1e-2 * len(rec_tr)
    A = rec_tr.T @ rec_tr + lam * np.eye(DIM, dtype=np.float64)
    B = rec_tr.T @ tru_tr
    W = np.linalg.solve(A, B).astype(np.float32)

    rec_te, tru_te = pool(test)
    return cosine(rec_te, tru_te), cosine(rec_te @ W, tru_te), W


def pq(train, test, m, bits=8, seed=0):
    """Product Quantization: m sub-spaces, 2^bits centroids each."""
    k = 1 << bits
    sub = DIM // m
    rec = np.empty_like(test)
    for j in range(m):
        sl = slice(j * sub, (j + 1) * sub)
        c = kmeans(train[:, sl], k, iters=15, seed=seed + j)
        lab = assign(test[:, sl], c)
        rec[:, sl] = c[lab]
    return cosine(rec, test)


def stats_row(name, c, bytes_per_row):
    ratio = DIM / bytes_per_row
    print(
        f"| {name:<32} | {bytes_per_row:5.1f} B | {ratio:4.1f}x | {c.mean():.4f} | {np.median(c):.4f} | {np.quantile(c, 0.05):.4f} |"
    )


def main():
    shard_path = sys.argv[1] if len(sys.argv) > 1 else find_default_shard()
    if not shard_path or not os.path.isfile(shard_path):
        print(f"Error: shard not found. Pass path to model-plefp8-00000.safetensors as argument.")
        sys.exit(1)

    t0 = time.time()
    x = load_safetensors_slice(shard_path)
    print(f"Loaded {len(x):,} rows, dimension={x.shape[1]}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x))
    ntr = int(len(x) * 0.6)
    train, test = x[idx[:ntr]], x[idx[ntr:]]
    print(f"Training set: {len(train):,} rows | Held-out evaluation: {len(test):,} rows\n")

    print("| Method                           | Bytes/Row | Ratio | Mean Cos | Med Cos | 5th Pct |")
    print("|:---------------------------------|:---------:|:-----:|:--------:|:-------:|:-------:|")

    # HashK R=4
    c_raw, c_ridge, W = hashk(train, test, R=4, seed=1)
    stats_row("HashK R=4 (mean-pool only)", c_raw, DIM / 4)
    stats_row("HashK R=4 (mean-pool + ridge W)", c_ridge, DIM / 4)

    # Product Quantization
    for m in (40, 20, 10, 8):
        c = pq(train[:120_000], test, m=m, seed=7)
        stats_row(f"PQ (m={m})", c, m)

    print(f"\nBenchmark completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
