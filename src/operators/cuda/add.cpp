#include "method.h"

extern void add(int n, const float *x, const float *y, float *z);

REGISTER_METHOD(CUDA, add, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  auto *out = ctx.output(0);
  out->resize(t1.shape(), t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *d2 = static_cast<const float *>(t2.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  auto N = t1.size();
  add(N, d1, d2, out_d);
});
