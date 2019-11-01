#include "method.h"

REGISTER_CPU_METHOD(sub, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  auto *out = ctx.output(0);
  out->resize(t1.shape(), t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *d2 = static_cast<const float *>(t2.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  for (auto i = 0; i < t1.size(); ++i) {
    out_d[i] = d1[i] - d2[i];
  }
});

REGISTER_GRAD(sub,
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                return {ginputs[0], call("neg", {ginputs[0]})[0]};
              });

REGISTER_SHAPE(sub,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
                 return { inputs[0]->shape };
               });
