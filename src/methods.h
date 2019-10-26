#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "error.h"
#include "tensor.h"
#include "variable.h"

#define ENFORCE(cond) \
if (!(cond)) { \
  throw std::runtime_error("Failed: "#cond);\
}

class Context {
 public:
  explicit Context(std::vector<const Tensor *> &inputs,
                   std::vector<Tensor *> &outputs)
      : inputs_(inputs), outputs_(outputs) {}
  const Tensor &input(int index);
  Tensor *output(int index);
  std::vector<Tensor *> outputs();

 private:
  std::vector<const Tensor *> &inputs_;
  std::vector<Tensor *> &outputs_;
};

using GradFn = std::function<std::vector<Variable*>(
      const std::vector<Variable*>&,
      const std::vector<Variable*>&
      )>;

using ShapeFn = std::function<std::vector<Size>(
      const std::vector<Variable*>&
      )>;

struct Method {
  std::string name;
  std::function<void(Context &ctx)> kernel;
  GradFn grad;
  ShapeFn shape;
  size_t num_outputs;
};

std::unordered_map<std::string, Method> &getMethodMap();
inline const Method& getMethod(std::string name) {
	auto method_iter = getMethodMap().find(name);
  HMA_ENFORCE(method_iter != getMethodMap().end());
	auto& method = method_iter->second;
	return method;
}

class RegMethod {
 public:
  RegMethod(std::string name, std::function<void(Context &ctx)> kernel,
            size_t num_out) {
    getMethodMap()[name].name = name;
    getMethodMap()[name].kernel = kernel;
    getMethodMap()[name].num_outputs = num_out;
  }
};

class RegGrad{
 public:
  RegGrad(std::string name, GradFn grad) {
    getMethodMap()[name].grad = grad;
  }
};

class RegShape{
 public:
  RegShape(std::string name, ShapeFn shape) {
    getMethodMap()[name].shape = shape;
  }
};

#define REG_W_NUM_OUTPUTS(name, kernel, num_outputs) \
  static RegMethod _reg_method_##name(#name, kernel, num_outputs);
#define REG_DEFAULT_OUTPUTS(name, kernel) \
  static RegMethod _reg_method_##name(#name, kernel, 1);

// Cool trick, right? Found on stackoverflow
#define GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4
#define REG_METHOD_MACRO_CHOOSER(...) \
  GET_4TH_ARG(__VA_ARGS__, REG_W_NUM_OUTPUTS, REG_DEFAULT_OUTPUTS)

#define REGISTER_METHOD(...) REG_METHOD_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

// ...'s are for commas in the macro invocations (with lambdas)
#define REGISTER_GRAD(name, ...) \
  static RegGrad _reg_grad_##name(#name, __VA_ARGS__);
#define REGISTER_SHAPE(name, ...) \
  static RegShape _reg_shape_##name(#name, __VA_ARGS__);
