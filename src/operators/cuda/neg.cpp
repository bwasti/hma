#include "method.h"

extern void neg(int n, const float *x, float *z);

REGISTER_METHOD(CUDA, neg, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  auto *out = ctx.output(0);
  out->resize(t1.shape(), t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  auto N = t1.size();
  neg(N, d1, out_d);
});
