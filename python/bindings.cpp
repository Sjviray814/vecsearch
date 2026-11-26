// pybind11 bindings: numpy in, numpy out. Batch search releases the GIL and runs multithreaded.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <thread>
#include <vector>

#include "vecsearch/hnsw.h"

namespace py = pybind11;
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Labels = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

static vs::Metric parse_metric(const std::string& m) {
    if (m == "l2") return vs::Metric::L2;
    if (m == "ip") return vs::Metric::InnerProduct;
    if (m == "cosine") return vs::Metric::Cosine;
    throw std::invalid_argument("metric must be 'l2', 'ip' or 'cosine'");
}

PYBIND11_MODULE(_vecsearch, m) {
    m.doc() = "HNSW approximate nearest-neighbor search";
    py::class_<vs::HNSWIndex>(m, "HNSWIndex")
        .def(py::init([](size_t dim, const std::string& metric, size_t M, size_t ef_construction, uint64_t seed) {
                 return vs::HNSWIndex(dim, parse_metric(metric), M, ef_construction, seed);
             }),
             py::arg("dim"), py::arg("metric") = "l2", py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("seed") = 42)
        .def("__len__", &vs::HNSWIndex::size)
        .def_property_readonly("dim", &vs::HNSWIndex::dim)
        .def("add",
             [](vs::HNSWIndex& idx, Array vecs, Labels labels) {
                 if (vecs.ndim() != 2 || size_t(vecs.shape(1)) != idx.dim())
                     throw std::invalid_argument("vectors must have shape (n, dim)");
                 if (labels.ndim() != 1 || labels.shape(0) != vecs.shape(0))
                     throw std::invalid_argument("labels must have shape (n,)");
                 py::gil_scoped_release release;
                 idx.add_batch(vecs.data(), labels.data(), size_t(vecs.shape(0)));
             },
             py::arg("vectors"), py::arg("labels"))
        .def("search",
             [](const vs::HNSWIndex& idx, Array queries, size_t k, size_t ef, unsigned threads) {
                 if (queries.ndim() != 2 || size_t(queries.shape(1)) != idx.dim())
                     throw std::invalid_argument("queries must have shape (nq, dim)");
                 const size_t nq = size_t(queries.shape(0));
                 py::array_t<int64_t> labels({nq, k});
                 py::array_t<float> dists({nq, k});
                 auto L = labels.mutable_unchecked<2>();
                 auto D = dists.mutable_unchecked<2>();
                 const float* q = queries.data();
                 if (threads == 0) threads = std::thread::hardware_concurrency();
                 {
                     py::gil_scoped_release release;
                     std::atomic<size_t> next{0};
                     std::vector<std::thread> pool;
                     for (unsigned t = 0; t < threads; ++t)
                         pool.emplace_back([&] {
                             for (size_t i; (i = next++) < nq;) {
                                 auto res = idx.search(q + i * idx.dim(), k, ef);
                                 for (size_t j = 0; j < k; ++j) {
                                     L(i, j) = j < res.size() ? int64_t(res[j].label) : -1;
                                     D(i, j) = j < res.size() ? res[j].distance : INFINITY;
                                 }
                             }
                         });
                     for (auto& th : pool) th.join();
                 }
                 return py::make_tuple(labels, dists);
             },
             py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64, py::arg("threads") = 0)
        .def("save", &vs::HNSWIndex::save, py::arg("path"))
        .def_static("load", &vs::HNSWIndex::load, py::arg("path"));
}
