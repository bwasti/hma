#pragma once

#include "error.h"
#include "tensor.h"

#include <list>
#include <vector>

struct Variable;
struct Tensor;

// Main API, call is a lazy invocation. debug_info optional
std::vector<Variable *> call(const std::string &,
                             const std::vector<Variable *> &,
                             std::string debug_info = "");
// Resolve evaluates the recorded operations and produces
// a real Tensor.
Tensor *resolve(const Variable *v);

// By default, call only records a maximum of DEFAULT_LAZINESS
// operations before resolving itself into cache.
#define DEFAULT_LAZINESS 10000
void setLaziness(size_t laziness);

// Implementation details below

struct Size {
  enum class Tag { Id, Number };

  Size(size_t i) : tag(Tag::Number), data(i) {}
  Size() : tag(Tag::Id), data(getNewId()) {}
  std::string str() const;

  size_t getNewId();

  Tag tag;
  size_t data;
};
using Shape = std::vector<Size>;

struct Method;
struct Graph;
struct Operator;
struct Variable {
  std::vector<Size> shape;
  Operator *op;
  std::vector<Operator *> deps;
  Graph *graph;
  mutable Tensor *tensor = nullptr;
  mutable size_t depth = 0;
};

struct Operator {
  Operator(std::string method_, const std::vector<Variable *> &inputs_,
           size_t num_outputs, Graph *graph_);
  const Method *method;
  std::vector<Variable *> inputs;
  std::vector<Variable *> outputs;
  Graph *graph;
};

struct Graph {
  inline Variable *create_var() {
    variables.emplace_back();
    auto *v = &variables.back();
    v->graph = this;
    return v;
  }

  inline Operator *create_op(std::string method,
                             const std::vector<Variable *> &inputs,
                             size_t num_outputs) {
    operators.emplace_back(method, inputs, num_outputs, this);
    auto *o = &operators.back();
    o->graph = this;
    return o;
  }

private:
  std::list<Variable> variables;
  std::list<Operator> operators;
};
