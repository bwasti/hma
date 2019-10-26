#include "methods.h"

std::unordered_map<std::string, Method> &getMethodMap() {
  static std::unordered_map<std::string, Method> methods_;
  return methods_;
}

const Tensor &Context::input(int index) { return *(inputs_[index]); }

Tensor *Context::output(int index) { return outputs_[index]; }

std::vector<Tensor *> Context::outputs() { return outputs_; }

