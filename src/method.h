#pragma once

#include "error.h"
#include "exec.h"
#include "tag.h"
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
  size_t num_inputs() const;
  Tensor *output(int index);
  std::vector<Tensor *> outputs();

private:
  std::vector<const Tensor *> &inputs_;
  std::vector<Tensor *> &outputs_;
};

using GradFn = std::function<std::vector<Variable *>(
    const std::vector<Variable *> &, const std::vector<Variable *> &)>;

using ShapeFn =
    std::function<std::vector<Shape>(const std::vector<Variable *> &)>;

struct Method {
  std::string name;
  std::vector<std::function<void(Context &ctx)>> kernels;
  GradFn grad;
  ShapeFn shape;
  size_t num_outputs;
};

std::unordered_map<std::string, Method> &getMethodMap();
inline const Method &getMethod(std::string name) {
  auto method_iter = getMethodMap().find(name);
  HMA_ENFORCE(method_iter != getMethodMap().end(), name);
  auto &method = method_iter->second;
  return method;
}

class RegMethod {
public:
  RegMethod(size_t tag, std::string name,
            std::function<void(Context &ctx)> kernel, size_t num_out=1) {
    auto &method = getMethodMap()[name];
    method.name = name;
    // This is kind of sketchy if `getTag` is used elsewhere
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

#define REGISTER_METHOD(tag, name, ...) \
  static RegMethod _reg_method_##name##_##tag(getTag(#tag), #name, __VA_ARGS__);
#define REGISTER_CPU_METHOD(...) REGISTER_METHOD(CPU, __VA_ARGS__)

// ...'s are for commas in the macro invocations (with lambdas)
#define REGISTER_GRAD(name, ...)                                               \
  static RegGrad _reg_grad_##name(#name, __VA_ARGS__);
#define REGISTER_SHAPE(name, ...)                                              \
  static RegShape _reg_shape_##name(#name, __VA_ARGS__);

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2> &pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};
std::unordered_map<std::pair<size_t, size_t>, std::string, pair_hash>& getTagPairMap();

class RegTagPair {
public:
  RegTagPair(size_t tag_from, size_t tag_to, std::string name) {
    getTagPairMap()[std::make_pair(tag_from, tag_to)] = name;
  }
};

#define REGISTER_TAG_PAIR(from, to, name) \
  static RegTagPair _reg_tag_pair_##name##_##from##_##to(getTag(#from), getTag(#to), #name);

const std::function<void(Context &ctx)>& tagPairMethod(size_t from, size_t to);
