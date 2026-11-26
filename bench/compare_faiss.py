"""Compare vecsearch against FAISS IndexHNSWFlat on SIFT1M (or any .fvecs/.ivecs set).

    python bench/compare_faiss.py data/sift [--threads N]

Both indexes use M=16, efConstruction=200, L2, and are scored against the dataset's official
ground truth (sift_groundtruth.ivecs). QPS is batch search over all 10K queries.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import _vecsearch as vs  # noqa: E402


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(dtype)


def recall_at_10(found, truth):
    hits = sum(len(set(f[:10]) & set(t[:10])) for f, t in zip(found, truth))
    return hits / (len(truth) * 10)


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--skip-faiss", action="store_true")
    args = ap.parse_args()

    base = read_vecs(os.path.join(args.dir, "sift_base.fvecs"), np.float32)
    queries = read_vecs(os.path.join(args.dir, "sift_query.fvecs"), np.float32)
    truth = read_vecs(os.path.join(args.dir, "sift_groundtruth.ivecs"), np.int32)
    n, d = base.shape
    print(f"base {n}x{d}, queries {len(queries)}, threads {args.threads}")

    idx = vs.HNSWIndex(d, "l2", M=16, ef_construction=200)
    _, t_vs = timed(lambda: idx.add(base, np.arange(n, dtype=np.uint64)))
    print(f"vecsearch build: {t_vs:.1f}s (single-threaded)")

    fidx = None
    if not args.skip_faiss:
        import faiss

        faiss.omp_set_num_threads(1)  # match vecsearch's single-threaded build
        fidx = faiss.IndexHNSWFlat(d, 16)
        fidx.hnsw.efConstruction = 200
        _, t_f = timed(lambda: fidx.add(base))
        print(f"faiss build:     {t_f:.1f}s (single-threaded)")
        faiss.omp_set_num_threads(args.threads)

    print(f"\n{'ef':>5} {'vs recall':>10} {'vs QPS':>10} {'faiss recall':>13} {'faiss QPS':>10}")
    rows = []
    for ef in (16, 32, 64, 128, 256, 512):
        (labels, _), t = timed(lambda: idx.search(queries, k=10, ef=ef, threads=args.threads))
        r_vs, q_vs = recall_at_10(labels, truth), len(queries) / t
        r_f = q_f = float("nan")
        if fidx is not None:
            fidx.hnsw.efSearch = ef
            (_, flabels), t = timed(lambda: fidx.search(queries, 10))
            r_f, q_f = recall_at_10(flabels, truth), len(queries) / t
        rows.append((ef, r_vs, q_vs, r_f, q_f))
        print(f"{ef:>5} {r_vs:>10.4f} {q_vs:>10.0f} {r_f:>13.4f} {q_f:>10.0f}")


if __name__ == "__main__":
    main()
