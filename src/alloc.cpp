#include "alloc.h"

std::unordered_map<size_t, void *(*)(size_t)> &getAllocMap() {
  static std::unordered_map<size_t, void *(*)(size_t)> map_;
  return map_;
}
std::unordered_map<size_t, void (*)(void *)> &getDeallocMap() {
  static std::unordered_map<size_t, void (*)(void *)> map_;
  return map_;
}

REGISTER_ALLOC(CPU, malloc, free);
