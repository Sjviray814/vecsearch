# vecsearch

An HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor index written from
scratch in C++17, with SIMD distance kernels and Python bindings.

- `include/vecsearch/hnsw.h`: the index (header-only). Covers insert, beam search, the Algorithm 4
  neighbor-selection heuristic, L2/inner-product/cosine metrics, save/load, and exact brute-force search
- `include/vecsearch/distance.h`: ARM NEON and AVX2+FMA kernels, plus a scalar fallback
- `python/bindings.cpp`: pybind11 module. Takes numpy in and returns numpy out; batch search releases the GIL and runs multithreaded
- `bench/bench.cpp`: recall@10 and QPS sweep across `ef`, either synthetic or on `.fvecs` (SIFT1M)
- `tests/test_hnsw.cpp`: SIMD vs. scalar, edge cases, recall floors, save/load round-trip

## Build

```bash
python3 -m venv .venv && .venv/bin/pip install pybind11 numpy
cmake -S . -B build -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build -j
./build/test_hnsw
./build/bench 100000 128 1000
```

```python
import sys; sys.path.insert(0, "build")
import numpy as np, _vecsearch as vs
idx = vs.HNSWIndex(dim=384, metric="cosine")
idx.add(vectors.astype("float32"), np.arange(len(vectors), dtype="uint64"))
labels, dists = idx.search(queries, k=10, ef=128)
```

## Results (Apple M2, 8 threads; 100K × 128-d clustered Gaussian, M=16, efC=200)

| ef | recall@10 | QPS (1 thread) | QPS (8 threads) | vs. exact |
|---:|---:|---:|---:|---:|
| 32 | 0.926 | 13.7K | 64.2K | 36× |
| 64 | 0.984 | 8.9K | 44.9K | 25× |
| 128 | 0.999 | 5.4K | 33.1K | 19× |
| 256 | 1.000 | 3.6K | 23.1K | 13× |

Build time is 15.8s, about 6.3K inserts/s single-threaded.

## Roadmap

1. **Real-data benchmark**: SIFT1M (`bench --fvecs sift_base.fvecs sift_query.fvecs`), with
   FAISS `IndexHNSWFlat` side by side at equal recall.
2. **Parallel construction**: per-node locks so `add` scales across cores.
3. **Filtered search and deletes**: tombstones, plus a label-predicate during the beam search.
4. **Scalar quantization (int8)**: cut memory about 4×, then re-rank with full precision.
5. **RAG / semantic search demo**: embed a corpus (for example arXiv abstracts) with a local
   sentence-transformer, serve `/search` from FastAPI, and add a small web UI.
6. **CI**: GitHub Actions running tests on x86 (AVX2) and ARM (NEON).

Update the resume bullets as these land, using the numbers this repo measures.
