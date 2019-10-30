#include "method.h"

REGISTER_CPU_METHOD(broadcast, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  auto *out = ctx.output(0);
  out->resize(t2.shape(), t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  for (auto i = 0; i < out->size(); ++i) {
    out_d[i] = d1[0];
  }
});

REGISTER_GRAD(broadcast,
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                return call("sum", { ginputs[0] });
              });

REGISTER_SHAPE(broadcast,
               [](const std::vector<Variable *> &inputs) -> std::vector<Size> {
                 return inputs[1]->shape;
               });


