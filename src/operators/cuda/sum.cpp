#include "method.h"

extern void sum(int n, const float *x, float *z);

REGISTER_METHOD(CUDA, sum, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  auto *out = ctx.output(0);
  out->resize({1}, t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  auto N = t1.size();
  sum(N, d1, out_d);
});
