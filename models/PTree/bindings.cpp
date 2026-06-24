#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

#include "ptree.h"

using Mat    = Eigen::MatrixXd;
using Vec    = Eigen::VectorXd;
using Indice = Eigen::VectorXi;

template<typename T>
using Ref    = Eigen::Ref<T>;

namespace py = pybind11;

PYBIND11_MODULE(_ptree_cpp, m) {
    m.doc() = "_ptree_cpp";

    py::class_<ptree::PTree>(m, "PTree")
        .def(py::init<const Ref<const Mat>&, const Ref<const Indice>&, int, int, int, double>(),
             py::arg("value"),
             py::arg("offset"),
             py::arg("max_depth"),
             py::arg("q"),
             py::arg("min_sample_leaf"),
             py::arg("gamma") = 1e-4)
        .def(py::init<const Ref<const Mat>&, const Ref<const Indice>&, int, int, int, const Ref<const Vec>&, double>(),
             py::arg("value"),
             py::arg("offset"),
             py::arg("max_depth"),
             py::arg("q"),
             py::arg("min_sample_leaf"),
             py::arg("factor"),
             py::arg("gamma") = 1e-4)
        .def("grow", &ptree::PTree::grow, py::call_guard<py::gil_scoped_release>())
        .def("predict", &ptree::PTree::predict)
        .def_readonly("tree_weight", &ptree::PTree::tree_weight);
}