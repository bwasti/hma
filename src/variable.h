#pragma once

#include "tensor.h"

#include <list>
#include <vector>

struct Size {
  enum Tag {
    Id,
    Number
  };
  Tag tag;

  // union {
  int id;
  int num;
  // }
};

struct Variable;

std::vector<Variable*> call(const std::string&, const std::vector<Variable*>&);
Tensor* resolve(const Variable* v);

struct Method;
struct Graph;
struct Operator;
struct Variable {
  std::vector<Size> shape;
  Operator* op;
  std::vector<Operator*> deps;
  Graph* graph;
  mutable Tensor* tensor = nullptr;
  mutable size_t depth = 0;
};

struct Operator {
  Operator(std::string method_, const std::vector<Variable*>& inputs_, size_t num_outputs, Graph* graph_);
  const Method* method;
  std::vector<Variable*> inputs;
  std::vector<Variable*> outputs;
  Graph* graph;
};

struct Graph {
  inline Variable* create_var() {
		variables.emplace_back();
		auto* v = &variables.back();
		v->graph = this;
		return v;
	}

  inline Operator* create_op(std::string method, const std::vector<Variable*>& inputs, size_t num_outputs) {
		operators.emplace_back(method, inputs, num_outputs, this);
		auto* o = &operators.back();
		o->graph = this;
		return o;
	}
  
 private:
  std::list<Variable> variables;
  std::list<Operator> operators;
};
