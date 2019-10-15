#include "tensor.h"
#include <stdlib.h>

std::vector<size_t> Tensor::shape() const { return shape_; }
void *Tensor::ptr() { return ptr_; }
const void *Tensor::ptr() const { return ptr_; }
void *Tensor::release() {
  auto *ptr = ptr_;
  ptr_ = nullptr;
  return ptr;
}

size_t Tensor::size(size_t from) const {
  size_t s = 1;
  for (auto i = from; i < shape_.size(); ++i) {
    const auto &s_ = shape_[i];
    s *= s_;
  }
  return s;
}

Tensor::Dtype Tensor::dtype() const { return dtype_; }
size_t Tensor::dtype_size() const {
  switch (dtype_) {
    case Tensor::Dtype::float_:
      return sizeof(float);
    case Tensor::Dtype::int_:
      return sizeof(int);
  }
  return 0;
}
void Tensor::resize(const std::vector<size_t> &shape, Dtype d) {
  shape_ = shape;
  dtype_ = d;
  ptr_ = aligned_alloc(32, dtype_size() * size());
}
