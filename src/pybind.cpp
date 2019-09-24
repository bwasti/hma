#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
#include "methods.h"
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(hma, m) {
  m.def("from_numpy",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> arr)
            -> std::shared_ptr<Tensor> {
          py::buffer_info buf = arr.request();
          auto t = std::make_shared<Tensor>();
          t->resize(std::vector<size_t>{buf.shape.begin(), buf.shape.end()},
                    Tensor::Dtype::float_);
          memcpy(t->ptr(), buf.ptr, t->size() * t->dtype_size());
          return t;
        });

  m.def("to_numpy", [](std::shared_ptr<Tensor> t) -> py::array_t<float> {
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

  py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
      .def(py::init<std::vector<size_t>>())
      .def_property_readonly("shape", &Tensor::shape);

  for (const auto &method : getMethodMap()) {
    m.def(method.first.c_str(),
          [&method](std::vector<std::shared_ptr<Tensor>> owned_inputs)
              -> std::vector<std::shared_ptr<Tensor>> {
            std::vector<std::shared_ptr<Tensor>> owned_outputs;
            for (auto i = 0; i < method.second.num_outputs; ++i) {
              owned_outputs.emplace_back(std::make_shared<Tensor>());
            };

            std::vector<const Tensor *> inputs;
            std::vector<Tensor *> outputs;
            for (const auto &i : owned_inputs) {
              inputs.emplace_back(i.get());
            }
            for (const auto &i : owned_outputs) {
              outputs.emplace_back(i.get());
            }
            Context ctx{inputs, outputs};
            method.second.kernel(ctx);

            return owned_outputs;
          });
  }
}
