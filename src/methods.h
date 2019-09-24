#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensor.h"

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

struct Method {
  std::function<void(Context &ctx)> kernel;
  size_t num_outputs;
};
std::unordered_map<std::string, Method> &getMethodMap();

class RegMethod {
 public:
  RegMethod(std::string name, std::function<void(Context &ctx)> kernel,
            size_t num_out) {
    getMethodMap()[name] = Method{kernel, num_out};
  }
};

#define REG_W_NUM_OUTPUTS(name, kernel, num_outputs) \
  static RegMethod _reg_method_name(#name, kernel, num_outputs);
#define REG_DEFAULT_OUTPUTS(name, kernel) \
  static RegMethod _reg_method_name(#name, kernel, 1);

#define GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4
#define REG_METHOD_MACRO_CHOOSER(...) \
  GET_4TH_ARG(__VA_ARGS__, REG_W_NUM_OUTPUTS, REG_DEFAULT_OUTPUTS)

#define REGISTER_METHOD(...) REG_METHOD_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

//#define REGISTER_METHOD(name, kernel)
//  static RegMethod _reg_method_name(#name, kernel);
