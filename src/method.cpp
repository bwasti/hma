#include "method.h"

std::unordered_map<std::string, Method> &getMethodMap() {
  static std::unordered_map<std::string, Method> methods_;
  return methods_;
}

std::unordered_map<std::string, size_t> &getTagMap() {
  static std::unordered_map<std::string, size_t> tags_;
  return tags_;
}

size_t getTag(std::string tag_name) {
  if (getTagMap().find(tag_name) == getTagMap().end()) {
    getTagMap()[tag_name] = getTagMap().size();
  }
  return getTagMap().at(tag_name);
}

const Tensor &Context::input(int index) { return *(inputs_[index]); }

Tensor *Context::output(int index) { return outputs_[index]; }

std::vector<Tensor *> Context::outputs() { return outputs_; }
