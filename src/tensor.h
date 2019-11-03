#pragma once

#include "tag.h"
#include <iostream>
#include <vector>

class Tensor {
public:
  enum class Dtype {
    float_,
    int_,
    byte_,
  };
  Tensor() : tag_(getTag("CPU")) {}
  Tensor(size_t tag) : tag_(tag) {}
  Tensor(std::vector<size_t> shape) : shape_(shape) {}
  Tensor(std::vector<size_t> shape, void *ptr, Dtype d)
      : shape_(shape), ptr_(ptr), dtype_(d) {}
  ~Tensor();

  std::vector<size_t> shape() const;
  void resize(const std::vector<size_t> &, Dtype);
  size_t size(size_t from = 0) const;
  Dtype dtype() const;
  size_t dtype_size() const;
  size_t bytes() const;
  size_t tag() const;
  void setTag(size_t t);
  void *ptr();
  void *release();
  const void *ptr() const;

private:
  std::vector<size_t> shape_;
  void *ptr_;
  Dtype dtype_;
  bool requires_grad_;
  size_t tag_;
};
