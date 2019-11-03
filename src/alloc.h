#pragma once
#include <cstddef>
#include <unordered_map>

#include "tag.h"

// TODO typdef/using these fn sigs away
std::unordered_map<size_t, void *(*)(size_t)> &getAllocMap();
std::unordered_map<size_t, void (*)(void *)> &getDeallocMap();
std::unordered_map<size_t, void* (*)(void *, const void*, size_t)> &getMemcpyMap();

class RegAllocator {
public:
  RegAllocator(size_t tag, void *(*alloc)(size_t), void (*dealloc)(void *),
      void *(*memcpy)(void *, const void*, size_t)
      ) {
    getAllocMap()[tag] = alloc;
    getDeallocMap()[tag] = dealloc;
    getMemcpyMap()[tag] = memcpy;
  }
};

#define REGISTER_ALLOC(tag, alloc, dealloc, memcpy)                                    \
  static RegAllocator _reg_allocator_##tag(getTag(#tag), alloc, dealloc, memcpy);
