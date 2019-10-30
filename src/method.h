#pragma once

#include "error.h"
#include "tensor.h"
#include "variable.h"
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define ENFORCE(cond)                                                          \
  if (!(cond)) {                                                               \
    throw std::runtime_error("Failed: " #cond);                                \
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

using GradFn = std::function<std::vector<Variable *>(
    const std::vector<Variable *> &, const std::vector<Variable *> &)>;

using ShapeFn =
    std::function<std::vector<Size>(const std::vector<Variable *> &)>;

struct Method {
  std::string name;
  std::vector<std::function<void(Context &ctx)>> kernels;
  GradFn grad;
  ShapeFn shape;
  size_t num_outputs;
};

std::unordered_map<std::string, size_t> &getTagMap();
size_t getTag(std::string tag_name);

std::unordered_map<std::string, Method> &getMethodMap();
inline const Method &getMethod(std::string name) {
  auto method_iter = getMethodMap().find(name);
  HMA_ENFORCE(method_iter != getMethodMap().end());
  auto &method = method_iter->second;
  return method;
}

class RegMethod {
public:
  RegMethod(size_t tag, std::string name,
            std::function<void(Context &ctx)> kernel, size_t num_out) {
    auto &method = getMethodMap()[name];
    method.name = name;
    if (tag >= method.kernels.size()) {
      method.kernels.resize(tag + 1);
    }
    method.kernels[tag] = kernel;
    method.num_outputs = num_out;
  }
};

class RegGrad {
public:
  RegGrad(std::string name, GradFn grad) { getMethodMap()[name].grad = grad; }
};

class RegShape {
public:
  RegShape(std::string name, ShapeFn shape) {
    getMethodMap()[name].shape = shape;
  }
};

#define REG_W_NUM_OUTPUTS(tag, name, kernel, num_outputs)                      \
  static RegMethod _reg_method_##name_##tag(getTag(#tag), #name, kernel,       \
                                            num_outputs);
#define REG_DEFAULT_OUTPUTS(tag, name, kernel)                                 \
  static RegMethod _reg_method_##name_##tag(getTag(#tag), #name, kernel, 1);

// Cool trick, right? Found on stackoverflow
#define GET_5TH_ARG(arg1, arg2, arg3, arg4, arg5, ...) arg5
#define REG_METHOD_MACRO_CHOOSER(...)                                          \
  GET_5TH_ARG(__VA_ARGS__, REG_W_NUM_OUTPUTS, REG_DEFAULT_OUTPUTS)

#define REGISTER_METHOD(...) REG_METHOD_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define REGISTER_CPU_METHOD(...) REGISTER_METHOD(CPU, __VA_ARGS__)

// ...'s are for commas in the macro invocations (with lambdas)
#define REGISTER_GRAD(name, ...)                                               \
  static RegGrad _reg_grad_##name(#name, __VA_ARGS__);
#define REGISTER_SHAPE(name, ...)                                              \
  static RegShape _reg_shape_##name(#name, __VA_ARGS__);
