#include "method.h"

REGISTER_METHOD(neg, [](Context& ctx) {
  const auto &t = ctx.input(0);
  auto *out = ctx.output(0);
  out->resize(t.shape(), t.dtype());
  auto *d = static_cast<const float *>(t.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  for (auto i = 0; i < t.size(); ++i) {
    out_d[i] = -d[i];
  }
});

REGISTER_GRAD(neg,
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                return call("neg", {ginputs[0]});
              });

REGISTER_SHAPE(neg,
               [](const std::vector<Variable *> &inputs) -> std::vector<Size> {
                 return inputs[0]->shape;
               });

