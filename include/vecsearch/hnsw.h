#pragma once
// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016) for approximate
// nearest-neighbor search.
//
// Thread safety: search() is const and safe to call from many threads at once (each call borrows
// its own visited list from a pool). add() must not run concurrently with add() or search().
// TODO: per-node locks for parallel construction.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vecsearch/distance.h"

namespace vs {

enum class Metric : uint32_t { L2 = 0, InnerProduct = 1, Cosine = 2 };

struct SearchResult {
    float distance;
    uint64_t label;
};

class HNSWIndex {
public:
    HNSWIndex(size_t dim, Metric metric = Metric::L2, size_t M = 16, size_t ef_construction = 200,
              uint64_t seed = 42)
        : dim_(dim), metric_(metric), M_(M), M0_(2 * M), ef_construction_(ef_construction),
          level_mult_(1.0 / std::log(static_cast<double>(M))), rng_(seed) {
        if (dim == 0) throw std::invalid_argument("dim must be > 0");
        if (M < 2) throw std::invalid_argument("M must be >= 2");
    }

    size_t size() const { return labels_.size(); }
    size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }
    int max_level() const { return max_level_; }

    void reserve(size_t n) {
        data_.reserve(n * dim_);
        labels_.reserve(n);
        levels_.reserve(n);
        links_.reserve(n);
    }

    void add(const float* vec, uint64_t label) {
        const uint32_t id = static_cast<uint32_t>(labels_.size());
        data_.insert(data_.end(), vec, vec + dim_);
        if (metric_ == Metric::Cosine) normalize(&data_[size_t(id) * dim_]);
        labels_.push_back(label);
        const int level = random_level();
        levels_.push_back(level);
        links_.emplace_back(level + 1);
        visited_pool_.grow(labels_.size());

        if (entry_ < 0) {
            entry_ = static_cast<int>(id);
            max_level_ = level;
            return;
        }

        const float* q = vector(id);
        uint32_t ep = static_cast<uint32_t>(entry_);
        float ep_dist = distance(q, vector(ep));
        // Greedy descent through layers above the new node's level.
        for (int l = max_level_; l > level; --l) greedy_step(q, ep, ep_dist, l);

        for (int l = std::min(level, max_level_); l >= 0; --l) {
            auto candidates = search_layer(q, ep, ep_dist, ef_construction_, l);
            const size_t m_max = l == 0 ? M0_ : M_;
            std::vector<uint32_t> neighbors = select_neighbors(candidates, M_);
            links_[id][l] = neighbors;
            for (uint32_t n : neighbors) connect(n, id, l, m_max);
            // Closest candidate seeds the next layer down.
            ep = candidates.front().second;
            ep_dist = candidates.front().first;
        }

        if (level > max_level_) {
            max_level_ = level;
            entry_ = static_cast<int>(id);
        }
    }

    void add_batch(const float* vecs, const uint64_t* labels, size_t n) {
        reserve(size() + n);
        for (size_t i = 0; i < n; ++i) add(vecs + i * dim_, labels[i]);
    }

    std::vector<SearchResult> search(const float* query, size_t k, size_t ef = 64) const {
        std::vector<SearchResult> out;
        if (entry_ < 0 || k == 0) return out;
        std::vector<float> qbuf;
        const float* q = query;
        if (metric_ == Metric::Cosine) {
            qbuf.assign(query, query + dim_);
            normalize(qbuf.data());
            q = qbuf.data();
        }
        uint32_t ep = static_cast<uint32_t>(entry_);
        float ep_dist = distance(q, vector(ep));
        for (int l = max_level_; l > 0; --l) greedy_step(q, ep, ep_dist, l);
        auto candidates = search_layer(q, ep, ep_dist, std::max(ef, k), 0);
        const size_t n = std::min(k, candidates.size());
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.push_back({candidates[i].first, labels_[candidates[i].second]});
        return out;
    }

    // Exact k-NN by linear scan; the ground truth for recall measurements.
    std::vector<SearchResult> brute_force(const float* query, size_t k) const {
        std::vector<float> qbuf;
        const float* q = query;
        if (metric_ == Metric::Cosine) {
            qbuf.assign(query, query + dim_);
            normalize(qbuf.data());
            q = qbuf.data();
        }
        std::priority_queue<std::pair<float, uint32_t>> top;  // max-heap of the k best
        for (uint32_t i = 0; i < size(); ++i) {
            float d = distance(q, vector(i));
            if (top.size() < k) top.emplace(d, i);
            else if (d < top.top().first) { top.pop(); top.emplace(d, i); }
        }
        std::vector<SearchResult> out(top.size());
        for (size_t i = out.size(); i-- > 0; top.pop()) out[i] = {top.top().first, labels_[top.top().second]};
        return out;
    }

    // ---- persistence ----
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        write(f, kMagic); write(f, kVersion);
        write(f, uint64_t(dim_)); write(f, uint32_t(metric_)); write(f, uint64_t(M_));
        write(f, uint64_t(ef_construction_)); write(f, int32_t(entry_)); write(f, int32_t(max_level_));
        write(f, uint64_t(size()));
        f.write(reinterpret_cast<const char*>(data_.data()), std::streamsize(data_.size() * sizeof(float)));
        f.write(reinterpret_cast<const char*>(labels_.data()), std::streamsize(labels_.size() * sizeof(uint64_t)));
        for (size_t i = 0; i < size(); ++i) {
            write(f, int32_t(levels_[i]));
            for (const auto& layer : links_[i]) {
                write(f, uint32_t(layer.size()));
                f.write(reinterpret_cast<const char*>(layer.data()), std::streamsize(layer.size() * sizeof(uint32_t)));
            }
        }
        if (!f) throw std::runtime_error("write failed: " + path);
    }

    static HNSWIndex load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        if (read<uint32_t>(f) != kMagic) throw std::runtime_error("not a vecsearch index: " + path);
        if (read<uint32_t>(f) != kVersion) throw std::runtime_error("unsupported index version");
        size_t dim = read<uint64_t>(f);
        Metric metric = Metric(read<uint32_t>(f));
        size_t M = read<uint64_t>(f);
        size_t efc = read<uint64_t>(f);
        HNSWIndex idx(dim, metric, M, efc);
        idx.entry_ = read<int32_t>(f);
        idx.max_level_ = read<int32_t>(f);
        size_t n = read<uint64_t>(f);
        idx.data_.resize(n * dim);
        idx.labels_.resize(n);
        f.read(reinterpret_cast<char*>(idx.data_.data()), std::streamsize(n * dim * sizeof(float)));
        f.read(reinterpret_cast<char*>(idx.labels_.data()), std::streamsize(n * sizeof(uint64_t)));
        idx.levels_.resize(n);
        idx.links_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            idx.levels_[i] = read<int32_t>(f);
            idx.links_[i].resize(size_t(idx.levels_[i]) + 1);
            for (auto& layer : idx.links_[i]) {
                layer.resize(read<uint32_t>(f));
                f.read(reinterpret_cast<char*>(layer.data()), std::streamsize(layer.size() * sizeof(uint32_t)));
            }
        }
        if (!f) throw std::runtime_error("truncated index file: " + path);
        idx.visited_pool_.grow(n);
        return idx;
    }

private:
    using Cand = std::pair<float, uint32_t>;  // (distance, internal id)

    static constexpr uint32_t kMagic = 0x56534857;  // "VSHW"
    static constexpr uint32_t kVersion = 1;

    // Epoch-tagged visited set: clearing is O(1) by bumping the epoch.
    struct VisitedList {
        std::vector<uint16_t> tags;
        uint16_t epoch = 0;
        void reset(size_t n) {
            if (tags.size() < n) tags.resize(n, 0);
            if (++epoch == 0) { std::fill(tags.begin(), tags.end(), 0); epoch = 1; }
        }
        bool visit(uint32_t i) {
            if (tags[i] == epoch) return false;
            tags[i] = epoch;
            return true;
        }
    };

    class VisitedPool {
    public:
        VisitedPool() = default;
        VisitedPool(VisitedPool&& o) noexcept : free_(std::move(o.free_)), n_(o.n_) {}
        VisitedPool& operator=(VisitedPool&& o) noexcept { free_ = std::move(o.free_); n_ = o.n_; return *this; }
        void grow(size_t n) { n_ = n; }
        std::unique_ptr<VisitedList> acquire() const {
            std::unique_ptr<VisitedList> v;
            {
                std::lock_guard<std::mutex> g(mu_);
                if (!free_.empty()) { v = std::move(free_.back()); free_.pop_back(); }
            }
            if (!v) v = std::make_unique<VisitedList>();
            v->reset(n_);
            return v;
        }
        void release(std::unique_ptr<VisitedList> v) const {
            std::lock_guard<std::mutex> g(mu_);
            free_.push_back(std::move(v));
        }
    private:
        mutable std::mutex mu_;
        mutable std::vector<std::unique_ptr<VisitedList>> free_;
        size_t n_ = 0;
    };

    const float* vector(uint32_t id) const { return &data_[size_t(id) * dim_]; }

    float distance(const float* a, const float* b) const {
        switch (metric_) {
            case Metric::L2: return l2_sq(a, b, dim_);
            case Metric::InnerProduct: return -dot(a, b, dim_);
            case Metric::Cosine: return 1.f - dot(a, b, dim_);
        }
        return 0.f;
    }

    void normalize(float* v) const {
        float n = std::sqrt(dot(v, v, dim_));
        if (n > 0.f) for (size_t i = 0; i < dim_; ++i) v[i] /= n;
    }

    int random_level() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng_);
        if (r <= 0.0) r = 1e-12;
        return static_cast<int>(-std::log(r) * level_mult_);
    }

    void greedy_step(const float* q, uint32_t& ep, float& ep_dist, int level) const {
        bool changed = true;
        while (changed) {
            changed = false;
            for (uint32_t n : links_[ep][level]) {
                float d = distance(q, vector(n));
                if (d < ep_dist) { ep_dist = d; ep = n; changed = true; }
            }
        }
    }

    // Beam search on one layer. Returns up to ef candidates sorted by ascending distance.
    std::vector<Cand> search_layer(const float* q, uint32_t ep, float ep_dist, size_t ef, int level) const {
        auto visited = visited_pool_.acquire();
        std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> frontier;  // min-heap
        std::priority_queue<Cand> best;                                              // max-heap
        visited->visit(ep);
        frontier.emplace(ep_dist, ep);
        best.emplace(ep_dist, ep);
        while (!frontier.empty()) {
            auto [d, c] = frontier.top();
            if (d > best.top().first && best.size() >= ef) break;
            frontier.pop();
            for (uint32_t n : links_[c][level]) {
                if (!visited->visit(n)) continue;
                float dn = distance(q, vector(n));
                if (best.size() < ef || dn < best.top().first) {
                    frontier.emplace(dn, n);
                    best.emplace(dn, n);
                    if (best.size() > ef) best.pop();
                }
            }
        }
        visited_pool_.release(std::move(visited));
        std::vector<Cand> out(best.size());
        for (size_t i = out.size(); i-- > 0; best.pop()) out[i] = best.top();
        return out;
    }

    // Neighbor-selection heuristic (Algorithm 4): keep a candidate only if it is closer to the
    // query than to every neighbor already kept. Preserves long-range links between clusters.
    std::vector<uint32_t> select_neighbors(const std::vector<Cand>& sorted, size_t m) const {
        std::vector<uint32_t> kept;
        kept.reserve(m);
        for (const auto& [d, c] : sorted) {
            if (kept.size() >= m) break;
            bool good = true;
            for (uint32_t r : kept) {
                if (distance(vector(c), vector(r)) < d) { good = false; break; }
            }
            if (good) kept.push_back(c);
        }
        return kept;
    }

    void connect(uint32_t node, uint32_t new_id, int level, size_t m_max) {
        auto& nl = links_[node][level];
        if (nl.size() < m_max) { nl.push_back(new_id); return; }
        std::vector<Cand> cands;
        cands.reserve(nl.size() + 1);
        const float* v = vector(node);
        cands.emplace_back(distance(v, vector(new_id)), new_id);
        for (uint32_t n : nl) cands.emplace_back(distance(v, vector(n)), n);
        std::sort(cands.begin(), cands.end());
        nl = select_neighbors(cands, m_max);
    }

    template <class T> static void write(std::ofstream& f, T v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
    template <class T> static T read(std::ifstream& f) { T v{}; f.read(reinterpret_cast<char*>(&v), sizeof(T)); return v; }

    size_t dim_;
    Metric metric_;
    size_t M_, M0_, ef_construction_;
    double level_mult_;
    std::mt19937_64 rng_;

    std::vector<float> data_;                               // n * dim, contiguous
    std::vector<uint64_t> labels_;                          // internal id -> user label
    std::vector<int> levels_;
    std::vector<std::vector<std::vector<uint32_t>>> links_;  // [node][level] -> neighbors
    int entry_ = -1;
    int max_level_ = -1;
    VisitedPool visited_pool_;
};

}  // namespace vs
