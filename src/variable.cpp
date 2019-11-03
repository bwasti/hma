#include "variable.h"
#include "error.h"
#include "method.h"

#include <sstream>

Operator::Operator(std::string name, const std::vector<Variable *> &inputs_,
                   size_t num_outputs, Graph *graph_)
    : inputs(inputs_), graph(graph_) {
  method = &getMethod(name);
  size_t max_depth = 0;
  for (const auto &i : inputs) {
    max_depth = std::max(i->depth, max_depth);
  }

  for (auto i = 0; i < num_outputs; ++i) {
    auto *v = graph->create_var();
    v->op = this;
    v->depth = max_depth + 1;
    outputs.emplace_back(v);
  }
}

size_t Size::getNewId() {
  static size_t id = 1;
  return id++;
}

std::string Size::str() const {
  std::stringstream ss;
  HMA_ENFORCE(tag == Tag::Id);
  size_t k = data;
  while (k) {
    auto n = k % 26;
    ss << "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[n - 1];
    k /= 26;
  }
  return ss.str();
}

