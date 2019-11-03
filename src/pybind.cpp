#include "error.h"
#include "grad.h"
#include "method.h"
#include "tensor.h"
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

// We use PyBind's lifetime management
// to deal with everything.
// Invariant: variable is in graph
struct TensorRef {
  Variable *variable;
  // TODO
  Graph *graph; // std::shared_ptr<Graph> graph;
};

static Graph g;
static bool is_debug = false;

PYBIND11_MODULE(hma, m) {
  py::class_<TensorRef, std::shared_ptr<TensorRef>>(m, "Tensor");
  py::class_<Size>(m, "Size")
      .def("__repr__",
           [](const Size &s) {
             if (s.tag == Size::Tag::Number) {
               std::stringstream ss;
               ss << s.data;
               return ss.str();
             }
             return s.str();
           })
      .def(py::init<int>())
      .def(py::init<>())
      .def("__int__", [](const Size &s) {
        HMA_ENFORCE(s.tag == Size::Tag::Number);
        return s.data;
      });

  m.def("set_debug", [](bool on) {
    if (on) {
      is_debug = true;
      setLaziness(0);
    } else {
      is_debug = true;
      setLaziness(DEFAULT_LAZINESS);
    }
  });
  m.def("debug", []() {
      return is_debug;
  });

  m.def("create_var", [](std::vector<Size> shape) {
    auto tr = std::make_shared<TensorRef>();
    tr->graph = &g; // std::make_shared<Graph>();
    auto *v = tr->graph->create_var();
    for (const auto &size : shape) {
      v->shape.emplace_back(size);
    }
    tr->variable = v;
    return tr;
  });

  m.def("from_numpy",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> arr)
            -> std::shared_ptr<TensorRef> {
          auto tr = std::make_shared<TensorRef>();
          tr->graph = &g; // std::make_shared<Graph>();
          auto *v = tr->graph->create_var();
          tr->variable = v;
          v->tensor = new Tensor();
          py::buffer_info buf = arr.request();
          for (const auto size : buf.shape) {
            v->shape.emplace_back(Size(size));
          }
          auto *t = v->tensor;
          t->resize(std::vector<size_t>{buf.shape.begin(), buf.shape.end()},
                    Tensor::Dtype::float_);
          for (size_t i = 0; i < buf.ndim; ++i) {
            const auto stride = buf.strides[i];
            HMA_ENFORCE(stride == sizeof(float) * t->size(i + 1),
              std::string("Can't handle strided numpy arrays.  Use numpy.ascontiguousarray"));
          }
          memcpy(t->ptr(), buf.ptr, t->size() * t->dtype_size());
          return tr;
        });

  m.def("to_numpy", [](std::shared_ptr<TensorRef> tr) -> py::array_t<float> {
    auto t = resolve(tr->variable);
    HMA_ENFORCE(t->tag() == getTag("CPU"));
    py::buffer_info buf;
    buf.ptr = (void *)t->ptr();
    buf.itemsize = sizeof(float);
    buf.format = py::format_descriptor<float>::format();
    buf.ndim = t->shape().size();
    buf.shape.resize(buf.ndim);
    buf.strides.resize(buf.ndim);
    for (auto i = 0; i < buf.ndim; ++i) {
      buf.shape[i] = t->shape()[i];
      buf.strides[i] = sizeof(float) * t->size(i + 1);
    }
    return py::array_t<float>(buf);
  });

  m.def("from_scalar", [](float val)
            -> std::shared_ptr<TensorRef> {
          auto tr = std::make_shared<TensorRef>();
          tr->graph = &g; // std::make_shared<Graph>();
          auto *v = tr->graph->create_var();
          tr->variable = v;
          v->tensor = new Tensor();
          v->shape.emplace_back(Size(1));
          auto *t = v->tensor;
          t->resize(std::vector<size_t>{1}, Tensor::Dtype::float_);
          ((float*)t->ptr())[0] = val;
          return tr;
        });

  m.def("to_scalar", [](std::shared_ptr<TensorRef> tr) -> float {
    auto t = resolve(tr->variable);
    HMA_ENFORCE(t->tag() == getTag("CPU"));
    return ((float*)t->ptr())[0];
  });

  m.def("get_tag", [](std::shared_ptr<TensorRef> tr) -> size_t {
    auto t = resolve(tr->variable);
    return t->tag();
  });

  m.def("get_tag", [](std::string s) -> size_t { return getTag(s); });

  // Return a tuple to make shapes hashable
  m.def("get_shape", [](std::shared_ptr<TensorRef> tr) {
          const auto& shape = tr->variable->shape;
          auto ret = py::tuple(shape.size());
          for (auto i = 0; i < shape.size(); ++i) {
            ret[i] = shape[i];
          }
          return ret;
        });

  for (const auto &method : getMethodMap()) {
    m.def(method.first.c_str(),
          [&method](std::vector<std::shared_ptr<TensorRef>> inputs, std::string debug_info)
              -> std::vector<std::shared_ptr<TensorRef>> {
            std::vector<Variable *> inputs_;
            auto graph = inputs[0]->graph;
            for (const auto &i : inputs) {
              inputs_.emplace_back(i->variable);
              // TODO: merge graphs, for memory opt
              HMA_ENFORCE(i->graph == graph);
            }
            std::vector<Variable *> vs = call(method.first, inputs_, debug_info);
            std::vector<std::shared_ptr<TensorRef>> out;
            for (const auto &v : vs) {
              auto tr = std::make_shared<TensorRef>();
              tr->variable = v;
              tr->graph = inputs[0]->graph;
              out.emplace_back(std::move(tr));
            }
            return out;
          }, py::arg("inputs"), py::arg("debug_info") = "");
  }

  m.def("grad", [](std::shared_ptr<TensorRef> y, std::shared_ptr<TensorRef> x,
                   std::shared_ptr<TensorRef> j) -> std::shared_ptr<TensorRef> {
    auto tr = std::make_shared<TensorRef>();
    tr->variable = grad(y->variable, x->variable, j->variable);
    tr->graph = tr->variable->graph;
    return tr;
  });

  m.def("resolve", [](std::shared_ptr<TensorRef> x) -> void {
    resolve(x->variable);
  });
}
