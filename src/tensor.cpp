#include "tensor.h"
#include "alloc.h"
#include "error.h"
#include <sstream>

std::vector<size_t> Tensor::shape() const { return shape_; }
void *Tensor::ptr() { return ptr_; }
const void *Tensor::ptr() const { return ptr_; }
void *Tensor::release() {
  auto *ptr = ptr_;
  ptr_ = nullptr;
  return ptr;
}
size_t Tensor::tag() const { return tag_; }
void Tensor::setTag(size_t tag) { tag_ = tag; }

size_t Tensor::size(size_t from) const {
  size_t s = 1;
  for (auto i = from; i < shape_.size(); ++i) {
    const auto &s_ = shape_[i];
    s *= s_;
  }
  return s;
}

size_t Tensor::bytes() const { return dtype_size() * size(); }

Tensor::Dtype Tensor::dtype() const { return dtype_; }
size_t Tensor::dtype_size() const {
  switch (dtype_) {
  case Tensor::Dtype::float_:
    return sizeof(float);
  case Tensor::Dtype::int_:
    return sizeof(int);
  case Tensor::Dtype::byte_:
    return sizeof(char);
  }
  return 0;
}
void Tensor::resize(const std::vector<size_t> &shape, Dtype d) {
  shape_ = shape;
  dtype_ = d;
  auto alloc = getAllocMap().at(tag_);
  if (!alloc) {
    std::stringstream ss;
    ss << "couldn't find allocator for tag "
      << getTagName(tag_);
    HMA_ENFORCE(alloc, ss.str());
  }
  ptr_ = alloc(bytes());
}

Tensor::~Tensor() {
  if (ptr_) {
    getDeallocMap().at(tag_)(ptr_);
  }
}
