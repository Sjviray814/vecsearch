// Minimal self-contained tests (no framework dependency). Exit code 0 = all passed.
#include <cassert>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "vecsearch/distance.h"
#include "vecsearch/hnsw.h"

static int failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static std::vector<float> random_vectors(size_t n, size_t d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> v(n * d);
    for (auto& x : v) x = g(rng);
    return v;
}

static double recall_at_k(const vs::HNSWIndex& idx, const std::vector<float>& queries, size_t nq,
                          size_t k, size_t ef) {
    size_t hit = 0;
    for (size_t i = 0; i < nq; ++i) {
        const float* q = &queries[i * idx.dim()];
        std::set<uint64_t> truth;
        for (auto& r : idx.brute_force(q, k)) truth.insert(r.label);
        for (auto& r : idx.search(q, k, ef)) hit += truth.count(r.label);
    }
    return double(hit) / double(nq * k);
}

static void test_simd_matches_scalar() {
    for (size_t d : {1, 7, 8, 15, 128, 129}) {
        auto a = random_vectors(1, d, 1), b = random_vectors(1, d, 2);
        float l2 = vs::l2_sq(a.data(), b.data(), d), l2s = vs::l2_sq_scalar(a.data(), b.data(), d);
        float dp = vs::dot(a.data(), b.data(), d), dps = vs::dot_scalar(a.data(), b.data(), d);
        CHECK(std::abs(l2 - l2s) <= 1e-3f * (1.f + std::abs(l2s)));
        CHECK(std::abs(dp - dps) <= 1e-3f * (1.f + std::abs(dps)));
    }
}

static void test_empty_and_single() {
    vs::HNSWIndex idx(4);
    float q[4] = {0, 0, 0, 0};
    CHECK(idx.search(q, 5).empty());
    float v[4] = {1, 2, 3, 4};
    idx.add(v, 99);
    auto r = idx.search(q, 5);
    CHECK(r.size() == 1 && r[0].label == 99);
}

static void test_exact_match_found() {
    const size_t n = 2000, d = 32;
    auto data = random_vectors(n, d, 3);
    vs::HNSWIndex idx(d);
    for (size_t i = 0; i < n; ++i) idx.add(&data[i * d], i);
    size_t found = 0;
    for (size_t i = 0; i < n; i += 10) {
        auto r = idx.search(&data[i * d], 1, 32);
        found += (!r.empty() && r[0].label == i);
    }
    CHECK(found == n / 10);
}

static void test_recall_l2() {
    const size_t n = 10000, d = 64, nq = 200;
    auto data = random_vectors(n, d, 4), queries = random_vectors(nq, d, 5);
    vs::HNSWIndex idx(d, vs::Metric::L2, 16, 200);
    for (size_t i = 0; i < n; ++i) idx.add(&data[i * d], i);
    double r = recall_at_k(idx, queries, nq, 10, 128);
    std::printf("  recall@10 (L2, n=%zu, d=%zu, ef=128) = %.4f\n", n, d, r);
    CHECK(r >= 0.95);
}

static void test_recall_cosine() {
    const size_t n = 5000, d = 48, nq = 100;
    auto data = random_vectors(n, d, 6), queries = random_vectors(nq, d, 7);
    vs::HNSWIndex idx(d, vs::Metric::Cosine);
    for (size_t i = 0; i < n; ++i) idx.add(&data[i * d], i);
    double r = recall_at_k(idx, queries, nq, 10, 128);
    std::printf("  recall@10 (cosine) = %.4f\n", r);
    CHECK(r >= 0.95);
    // Cosine ignores magnitude.
    std::vector<float> scaled(data.begin(), data.begin() + d);
    for (auto& x : scaled) x *= 37.f;
    auto res = idx.search(scaled.data(), 1, 64);
    CHECK(!res.empty() && res[0].label == 0);
}

static void test_save_load_roundtrip() {
    const size_t n = 3000, d = 16;
    auto data = random_vectors(n, d, 8), queries = random_vectors(50, d, 9);
    vs::HNSWIndex idx(d, vs::Metric::InnerProduct);
    for (size_t i = 0; i < n; ++i) idx.add(&data[i * d], 1000 + i);
    const char* path = "/tmp/vecsearch_test.idx";
    idx.save(path);
    auto loaded = vs::HNSWIndex::load(path);
    CHECK(loaded.size() == n);
    CHECK(loaded.metric() == vs::Metric::InnerProduct);
    for (size_t i = 0; i < 50; ++i) {
        auto a = idx.search(&queries[i * d], 10, 64), b = loaded.search(&queries[i * d], 10, 64);
        CHECK(a.size() == b.size());
        for (size_t j = 0; j < a.size() && j < b.size(); ++j) CHECK(a[j].label == b[j].label);
    }
    std::remove(path);
}

int main() {
    test_simd_matches_scalar();
    test_empty_and_single();
    test_exact_match_found();
    test_recall_l2();
    test_recall_cosine();
    test_save_load_roundtrip();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
