#include "method.h"
#include <sstream>

std::unordered_map<std::string, Method> &getMethodMap() {
  static std::unordered_map<std::string, Method> methods_;
  return methods_;
}

const Tensor &Context::input(int index) { return *(inputs_[index]); }


size_t Context::num_inputs() const {
  return inputs_.size();
}

Tensor *Context::output(int index) { return outputs_[index]; }

std::vector<Tensor *> Context::outputs() { return outputs_; }

std::unordered_map<std::pair<size_t, size_t>, std::string, pair_hash>& getTagPairMap() {
  static std::unordered_map<std::pair<size_t, size_t>, std::string, pair_hash> tag_to_map_;
  return tag_to_map_;
}
const std::function<void(Context &ctx)>& tagPairMethod(size_t from, size_t to) {
  auto name_iter = getTagPairMap().find(std::make_pair(from, to));
  if (name_iter == getTagPairMap().end()) {
    std::stringstream ss;
    ss << "Cannot convert between tag pair " << getTagName(from)
       << " -> " << getTagName(to);
    HMA_ENFORCE(name_iter != getTagPairMap().end(), ss.str())
  }
  auto name = name_iter->second;
  auto method_iter = getMethodMap().find(name);
  HMA_ENFORCE(method_iter != getMethodMap().end());
  const auto& to_method = method_iter->second.kernels[from];
  HMA_ENFORCE(to_method);
  return to_method;
}
