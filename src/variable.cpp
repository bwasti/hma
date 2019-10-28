#include "variable.h"
#include "error.h"
#include "method.h"

Operator::Operator(std::string name, const std::vector<Variable*>& inputs_,
                   size_t num_outputs, Graph* graph_)
    : inputs(inputs_), graph(graph_) {
  method = &getMethod(name);
  size_t max_depth = 0;
  for (const auto& i : inputs) {
    max_depth = std::max(i->depth, max_depth);
  }

  for (auto i = 0; i < num_outputs; ++i) {
    auto* v = graph->create_var();
    v->op = this;
    v->depth = max_depth + 1;
    outputs.emplace_back(v);
  }
}

void Variable::invalidate_deps() const {
  std::vector<const Variable*> frontier;
  for (auto dep : deps) {
    for (auto output : dep->outputs) {
      frontier.emplace_back(output);
    }
  }
  std::vector<const Variable*> next_frontier;
  while (frontier.size()) {
    next_frontier.clear();
    for (auto var : frontier) {
      if (var->tensor) {
        delete var->tensor;
      }
      for (auto dep : var->deps) {
        for (auto output : dep->outputs) {
          next_frontier.emplace_back(output);
        }
      }
    }
    frontier = next_frontier;
  }
}

void Variable::swap(Variable* v) {
  HMA_ENFORCE(v->graph == graph);
  invalidate_deps();
  auto old_t = tensor;
  tensor = v->tensor;
  v->tensor = old_t;
}

std::vector<Variable*> call(const std::string& name,
                            const std::vector<Variable*>& vs) {
  HMA_ENFORCE(vs.size());
  auto* graph = vs[0]->graph;
  for (const auto& v : vs) {
    HMA_ENFORCE(graph == v->graph);
    if (v->depth > 10000) {
      v->tensor = resolve(v);
    }
  }
  const auto& method = getMethod(name);
  auto op = graph->create_op(name, vs, method.num_outputs);
  for (auto& v : vs) {
    v->deps.emplace_back(op);
  }
  return op->outputs;
}

Tensor* resolve(const Variable* v) {
  if (v->tensor) {
    v->depth = 0;
    return v->tensor;
  } else {
    HMA_ENFORCE(v->op);
    std::vector<const Tensor*> inputs;
    for (const auto& i : v->op->inputs) {
      inputs.emplace_back(resolve(i));
    }
    std::vector<Tensor*> outputs;
    for (const auto& o : v->op->outputs) {
      o->tensor = new Tensor();
      outputs.emplace_back(o->tensor);
    }
    auto& method = *v->op->method;
    Context ctx{inputs, outputs};
    method.kernel(ctx);
    v->depth = 0;
    return v->tensor;
  }
}