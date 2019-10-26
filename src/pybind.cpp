#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
#include <set>
#include <queue>
#include <unordered_map>
#include "methods.h"
#include "tensor.h"
#include "error.h"

namespace py = pybind11;

// We use PyBind's lifetime management
// to deal with everything.
// Invariant: variable is in graph
struct TensorRef {
  Variable* variable;
  // TODO
  Graph* graph;//std::shared_ptr<Graph> graph;
};

static Graph g;

PYBIND11_MODULE(hma, m) {

  m.def("from_numpy",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> arr)
            -> std::shared_ptr<TensorRef> {
          auto tr = std::make_shared<TensorRef>();
          tr->graph = &g;//std::make_shared<Graph>();
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

  for (const auto &method : getMethodMap()) {
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

  m.def("grad", [](std::shared_ptr<TensorRef> y,
                   std::shared_ptr<TensorRef> x,
                   std::shared_ptr<TensorRef> j) {
    std::unordered_map<Variable*, std::set<int>> need_grad;
		need_grad[y->variable] = {-1};
    std::unordered_map<Variable*, std::set<int>> no_grad;
    std::queue<std::pair<Variable*, int>> q;
    // Iterate from X, as most nets work this way
    q.push(std::make_pair(x->variable, -1));
    // q contains variables that haven't been
    // traversed.
    while (q.size()) {
      // Take a variable and try to find y,
      // "staying left" (first dep every time).
      //
      //   |
      //   v
      //  dep1  dep2
      //    \   /
      //     var
      //
      // Every time we "stay left," add the other deps to q
      // If we find y -- add the whole route to need_grad
      // If we can't find y -- add the whole route to no_grad
      Variable* var;
      int index;
      std::tie(var, index) = q.front();
      q.pop();
      std::unordered_map<Variable*, std::set<int>> route;
      route[var] = {index};

      while (var) {
        if (var == y->variable) {
          need_grad.insert(route.begin(), route.end());
          break;
        }
        // add to q
        std::vector<std::pair<Variable*, int>> next;
        for (auto dep : var->deps) {
          auto i = 0;
          for (auto inp : dep->inputs) {
            if (inp == var) {
              for (const auto& out : dep->outputs) {
                next.emplace_back(std::make_pair(out, i));
              }
            }
            i++;
          }
        }
        if (!next.size()) {
          no_grad.insert(route.begin(), route.end());
          break;
        }
        auto iter = next.begin();
        var = iter->first;
        route[var].insert(0);
        iter++;
        while (iter != next.end()) {
          q.push(*iter);
          iter++;
        }
      }
    }
    // Now calculate the gradients
    std::unordered_map<Variable*, Variable*> grad_map;
    // This is the input
    grad_map[y->variable] = j->variable;
    std::vector<Operator*> frontier{ y->variable->op };
    std::vector<Operator*> next_frontier;
    while (frontier.size()) {
			next_frontier.clear();
      for (const auto& op : frontier) {
				std::vector<Variable*> grad_inputs;
				for (const auto& op_out : op->outputs) {
					HMA_ENFORCE(need_grad.find(op_out) != need_grad.end());
					auto grad_inp_iter = grad_map.find(op_out);
					HMA_ENFORCE(grad_inp_iter != grad_map.end());
					grad_inputs.emplace_back(grad_inp_iter->second);
				}
        auto g_outs = op->method->grad(op->inputs, grad_inputs);
				for (auto i = 0; i < g_outs.size(); ++i) {
					auto input = op->inputs[i];
					if (need_grad.find(input) != need_grad.end()) {
						if (grad_map.find(input) != grad_map.end()) {
							grad_map[input] = call("add", {grad_map[input], g_outs[i]})[0];
						} else {
							grad_map[input] = g_outs[i];
						}
						if (input->op) {
							next_frontier.emplace_back(input->op);
						} else {
							HMA_ENFORCE(input == x->variable);
						}
					}
				}
      }
			frontier = next_frontier;
    }
		auto tr = std::make_shared<TensorRef>();
		tr->variable = grad_map[x->variable];
		tr->graph = tr->variable->graph;
		return tr;
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
  for (const auto &method : getMethodMap()) {
    m.def((method.first + "_").c_str(),
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

