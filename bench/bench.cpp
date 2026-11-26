// Recall vs. throughput benchmark.
//   bench [n] [dim] [queries] [threads]
// Data is a Gaussian mixture (clustered, like real embeddings) unless --fvecs files are given:
//   bench --fvecs base.fvecs query.fvecs [threads]   (e.g. SIFT1M from corpus-texmex.irisa.fr)
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "vecsearch/hnsw.h"

using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Centers come from a fixed seed so base and query sets share one distribution.
static std::vector<float> clustered(size_t n, size_t d, size_t clusters, uint64_t seed) {
    std::mt19937_64 crng(12345), rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> centers(clusters * d);
    for (auto& x : centers) x = g(crng) * 4.f;
    std::uniform_int_distribution<size_t> pick(0, clusters - 1);
    std::vector<float> v(n * d);
    for (size_t i = 0; i < n; ++i) {
        size_t c = pick(rng);
        for (size_t j = 0; j < d; ++j) v[i * d + j] = centers[c * d + j] + g(rng);
    }
    return v;
}

static std::vector<float> read_fvecs(const std::string& path, size_t& dim) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    std::vector<float> out;
    int32_t d;
    while (f.read(reinterpret_cast<char*>(&d), 4)) {
        dim = size_t(d);
        size_t off = out.size();
        out.resize(off + dim);
        f.read(reinterpret_cast<char*>(&out[off]), std::streamsize(dim * 4));
    }
    return out;
}

template <class F>
static void parallel_for(size_t n, unsigned threads, F f) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t)
        pool.emplace_back([&] { for (size_t i; (i = next++) < n;) f(i); });
    for (auto& th : pool) th.join();
}

int main(int argc, char** argv) {
    size_t n = 100000, d = 128, nq = 1000, dim = 0;
    unsigned threads = std::thread::hardware_concurrency();
    std::vector<float> base, queries;
    if (argc >= 4 && std::strcmp(argv[1], "--fvecs") == 0) {
        base = read_fvecs(argv[2], dim);
        queries = read_fvecs(argv[3], dim);
        d = dim;
        n = base.size() / d;
        nq = queries.size() / d;
        if (argc > 4) threads = unsigned(std::stoul(argv[4]));
    } else {
        if (argc > 1) n = std::stoul(argv[1]);
        if (argc > 2) d = std::stoul(argv[2]);
        if (argc > 3) nq = std::stoul(argv[3]);
        if (argc > 4) threads = unsigned(std::stoul(argv[4]));
        base = clustered(n, d, 100, 1);
        queries = clustered(nq, d, 100, 2);
    }
    const size_t k = 10;
    std::printf("n=%zu dim=%zu queries=%zu k=%zu threads=%u\n", n, d, nq, k, threads);

    vs::HNSWIndex idx(d, vs::Metric::L2, 16, 200);
    idx.reserve(n);
    auto t0 = Clock::now();
    for (size_t i = 0; i < n; ++i) idx.add(&base[i * d], i);
    auto t1 = Clock::now();
    std::printf("build: %.2fs (%.0f inserts/s), max level %d\n", secs(t0, t1), n / secs(t0, t1), idx.max_level());

    // Ground truth by exact scan (parallel).
    std::vector<std::set<uint64_t>> truth(nq);
    t0 = Clock::now();
    parallel_for(nq, threads, [&](size_t i) {
        for (auto& r : idx.brute_force(&queries[i * d], k)) truth[i].insert(r.label);
    });
    t1 = Clock::now();
    double brute_qps = nq / secs(t0, t1);
    std::printf("brute force: %.0f QPS (%u threads)\n\n", brute_qps, threads);

    std::printf("%6s %10s %12s %12s %10s\n", "ef", "recall@10", "QPS(1 thr)", "QPS(all)", "vs brute");
    for (size_t ef : {16, 32, 64, 128, 256, 512}) {
        std::vector<size_t> hits(nq);
        t0 = Clock::now();
        for (size_t i = 0; i < nq; ++i) idx.search(&queries[i * d], k, ef);
        t1 = Clock::now();
        double qps1 = nq / secs(t0, t1);
        t0 = Clock::now();
        parallel_for(nq, threads, [&](size_t i) {
            for (auto& r : idx.search(&queries[i * d], k, ef)) hits[i] += truth[i].count(r.label);
        });
        t1 = Clock::now();
        double qpsn = nq / secs(t0, t1);
        size_t total = 0;
        for (auto h : hits) total += h;
        std::printf("%6zu %10.4f %12.0f %12.0f %9.1fx\n", ef, double(total) / double(nq * k), qps1, qpsn,
                    qpsn / brute_qps);
    }
    return 0;
}
