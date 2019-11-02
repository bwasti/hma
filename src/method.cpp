#include "method.h"

std::unordered_map<std::string, Method> &getMethodMap() {
  static std::unordered_map<std::string, Method> methods_;
  return methods_;
}

const Tensor &Context::input(int index) { return *(inputs_[index]); }

Tensor *Context::output(int index) { return outputs_[index]; }

std::vector<Tensor *> Context::outputs() { return outputs_; }

std::unordered_map<std::pair<size_t, size_t>, std::string, pair_hash>& getTagToMap() {
  static std::unordered_map<std::pair<size_t, size_t>, std::string, pair_hash> tag_to_map_;
  return tag_to_map_;
}
const std::function<void(Context &ctx)>& tagToMethod(size_t from, size_t to) {
  auto name = getTagToMap()[std::make_pair(from, to)];
  return getMethodMap()[name].kernels[from];
}
