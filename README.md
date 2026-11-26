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

## Results

### SIFT1M vs. FAISS (Apple M2, 8 search threads)

1M base vectors, 10K queries, scored against the official ground truth. Both indexes use
M=16, efConstruction=200, L2, and a single-threaded build. Reproduce with
`python bench/compare_faiss.py data/sift`.

| ef | vecsearch recall@10 | vecsearch QPS | FAISS recall@10 | FAISS QPS |
|---:|---:|---:|---:|---:|
| 16 | 0.802 | 15.7K* | 0.808 | 49.0K |
| 32 | 0.904 | 28.3K | 0.911 | 56.6K |
| 64 | 0.964 | 27.9K | 0.968 | 31.7K |
| 128 | **0.990** | **15.8K** | 0.990 | 14.3K |
| 256 | 0.997 | 7.7K | 0.997 | 6.1K |
| 512 | 0.999 | 2.8K | 0.999 | 2.8K |

Build times: vecsearch 624s, FAISS 529s.

- Recall matches FAISS at every beam width, to within 0.7 points.
- At high recall (ef ≥ 128), vecsearch throughput is equal to or above FAISS's: 11% higher at 99% recall@10.
- At low ef, FAISS is up to 2× faster.

\* The first batch includes thread-pool and page-fault warm-up. Low-ef throughput is the next
optimization target: prefetching neighbor vectors and a flat, cache-friendly adjacency layout.

### Synthetic (100K × 128-d clustered Gaussian, `./build/bench 100000 128 1000`)

| ef | recall@10 | QPS (1 thread) | QPS (8 threads) | vs. exact |
|---:|---:|---:|---:|---:|
| 64 | 0.984 | 8.9K | 44.9K | 25× |
| 128 | 0.999 | 5.4K | 33.1K | 19× |

## Roadmap

CI already runs on x86 (AVX2) and ARM (NEON) via GitHub Actions.

1. **Low-ef throughput**: software prefetch of neighbor vectors, a flat adjacency array instead of
   nested vectors, and query warm-up, to close the gap with FAISS below 95% recall.
2. **Parallel construction**: per-node locks so `add` scales across cores.
3. **Filtered search and deletes**: tombstones, plus a label-predicate during the beam search.
4. **Scalar quantization (int8)**: cut memory about 4×, then re-rank with full precision.
5. **RAG / semantic search demo**: embed a corpus (for example arXiv abstracts) with a local
   sentence-transformer, serve `/search` from FastAPI, and add a small web UI.

Update the resume bullets as these land, using the numbers this repo measures.
