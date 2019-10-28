#include "method.h"

REGISTER_METHOD(div, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  auto *out = ctx.output(0);
  out->resize(t1.shape(), t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *d2 = static_cast<const float *>(t2.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  for (auto i = 0; i < t1.size(); ++i) {
    out_d[i] = d1[i] / d2[i];
  }
});

REGISTER_GRAD(div,
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                auto b_2 = call("mul", {inputs[1], inputs[1]})[0];
                auto a_div_b_2 = call("div", {inputs[0], b_2})[0];
                return {call("div", {ginputs[0], inputs[1]})[0],
                        call("neg", {a_div_b_2})[0]
                        };
              });

REGISTER_SHAPE(div,
               [](const std::vector<Variable *> &inputs) -> std::vector<Size> {
                 return inputs[0]->shape;
               });
