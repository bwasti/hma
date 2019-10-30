#include "method.h"

REGISTER_CPU_METHOD(sum, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  auto *out = ctx.output(0);
  out->resize({1}, t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  out_d[0] = 0;
  for (auto i = 0; i < t1.size(); ++i) {
    out_d[0] += d1[i];
  }
});

REGISTER_GRAD(sum,
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                return call("broadcast", {ginputs[0], inputs[0]});
                // auto bcast = call("broadcast", { ginputs[0] , inputs[0]
                // })[0]; return call("mul", { bcast, inputs[0] });
              });

REGISTER_SHAPE(sum,
               [](const std::vector<Variable *> &inputs) -> std::vector<Size> {
                 return {Size(1)};
               });
