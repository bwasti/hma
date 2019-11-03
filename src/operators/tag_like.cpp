#include "method.h"
#include "alloc.h"

void genericTagLike(Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  auto *out = ctx.output(0);
  out->setTag(t2.tag());
  out->resize(t1.shape(), t1.dtype());

  // TODO impl device-like elim
  if (t1.tag() == t2.tag()) {
    auto *d1 = static_cast<const float *>(t1.ptr());
    auto *out_d = static_cast<float *>(out->ptr());
    getMemcpyMap()[t1.tag()](out_d, d1, t1.bytes());
  } else {
    tagPairMethod(t1.tag(), t2.tag())(ctx);
  }
}

// Move input 0 to device input 1 is on
REGISTER_CPU_METHOD(tag_like, genericTagLike);

REGISTER_SHAPE(tag_like,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
                 return { inputs[0]->shape };
               });

REGISTER_GRAD(tag_like,
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                return {ginputs[0], ginputs[0]};
              });
