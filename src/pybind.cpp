#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
#include "error.h"
#include "grad.h"
#include "method.h"
#include "tensor.h"

namespace py = pybind11;

// We use PyBind's lifetime management
// to deal with everything.
// Invariant: variable is in graph
struct TensorRef {
  Variable* variable;
  // TODO
  Graph* graph;  // std::shared_ptr<Graph> graph;
};

static Graph g;

PYBIND11_MODULE(hma, m) {
  m.def("from_numpy",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> arr)
            -> std::shared_ptr<TensorRef> {
          auto tr = std::make_shared<TensorRef>();
          tr->graph = &g;  // std::make_shared<Graph>();
          auto* v = tr->graph->create_var();
          tr->variable = v;
          v->tensor = new Tensor();
          py::buffer_info buf = arr.request();
          auto* t = v->tensor;
          t->resize(std::vector<size_t>{buf.shape.begin(), buf.shape.end()},
                    Tensor::Dtype::float_);
          memcpy(t->ptr(), buf.ptr, t->size() * t->dtype_size());
          return tr;
        });

  m.def("to_numpy", [](std::shared_ptr<TensorRef> tr) -> py::array_t<float> {
    auto t = resolve(tr->variable);
    py::buffer_info buf;
    buf.ptr = (void*)t->ptr();
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

  py::class_<TensorRef, std::shared_ptr<TensorRef>>(m, "Tensor");

  for (const auto& method : getMethodMap()) {
    m.def(method.first.c_str(),
          [&method](std::vector<std::shared_ptr<TensorRef>> inputs)
              -> std::vector<std::shared_ptr<TensorRef>> {
            std::vector<Variable*> inputs_;
            auto graph = inputs[0]->graph;
            for (const auto& i : inputs) {
              inputs_.emplace_back(i->variable);
              // TODO: merge graphs, for memory opt
              HMA_ENFORCE(i->graph == graph);
            }
            std::vector<Variable*> vs = call(method.first, inputs_);
            std::vector<std::shared_ptr<TensorRef>> out;
            for (const auto& v : vs) {
              auto tr = std::make_shared<TensorRef>();
              tr->variable = v;
              tr->graph = inputs[0]->graph;
              out.emplace_back(std::move(tr));
            }
            return out;
          });
  }

  m.def("grad", [](std::shared_ptr<TensorRef> y, std::shared_ptr<TensorRef> x,
                   std::shared_ptr<TensorRef> j) {
    auto tr = std::make_shared<TensorRef>();
    tr->variable = grad(y->variable, x->variable, j->variable);
    tr->graph = tr->variable->graph;
    return tr;
  });

  m.def("swap", [](std::shared_ptr<TensorRef> y, std::shared_ptr<TensorRef> x) {
    // TODO enforce shape info stuff
    y->variable->swap(x->variable);
  });

  // Old API
  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "TensorRaw")
      .def(py::init<std::vector<size_t>>())
      .def_property_readonly("shape", &Tensor::shape);

  m.def("from_numpy_",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> arr)
            -> std::shared_ptr<Tensor> {
          py::buffer_info buf = arr.request();
          auto t = std::make_shared<Tensor>();
          t->resize(std::vector<size_t>{buf.shape.begin(), buf.shape.end()},
                    Tensor::Dtype::float_);
          memcpy(t->ptr(), buf.ptr, t->size() * t->dtype_size());
          return t;
        });

  m.def("to_numpy_", [](std::shared_ptr<Tensor> t) -> py::array_t<float> {
    py::buffer_info buf;
    buf.ptr = t->ptr();
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
  for (const auto& method : getMethodMap()) {
    m.def((method.first + "_").c_str(),
          [&method](std::vector<std::shared_ptr<Tensor>> owned_inputs)
              -> std::vector<std::shared_ptr<Tensor>> {
            std::vector<std::shared_ptr<Tensor>> owned_outputs;
            for (auto i = 0; i < method.second.num_outputs; ++i) {
              owned_outputs.emplace_back(std::make_shared<Tensor>());
            };

            std::vector<const Tensor*> inputs;
            std::vector<Tensor*> outputs;
            for (const auto& i : owned_inputs) {
              inputs.emplace_back(i.get());
            }
            for (const auto& i : owned_outputs) {
              outputs.emplace_back(i.get());
            }
            Context ctx{inputs, outputs};
            method.second.kernel(ctx);

            return owned_outputs;
          });
  }
}
