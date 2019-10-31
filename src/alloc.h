#pragma once
#include <cstddef>
#include <unordered_map>

#include "tag.h"

std::unordered_map<size_t, void *(*)(size_t)> &getAllocMap();
std::unordered_map<size_t, void (*)(void *)> &getDeallocMap();

class RegAllocator {
public:
  RegAllocator(size_t tag, void *(*alloc)(size_t), void (*dealloc)(void *)) {
    getAllocMap()[tag] = alloc;
    getDeallocMap()[tag] = dealloc;
  }
};

#define REGISTER_ALLOC(tag, alloc, dealloc)                                    \
  static RegAllocator _reg_allocator_##tag(getTag(#tag), alloc, dealloc);
