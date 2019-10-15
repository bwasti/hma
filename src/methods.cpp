#include "methods.h"

std::unordered_map<std::string, Method> &getMethodMap() {
  static std::unordered_map<std::string, Method> methods_;
  return methods_;
}

const Tensor &Context::input(int index) { return *(inputs_[index]); }

Tensor *Context::output(int index) { return outputs_[index]; }

std::vector<Tensor *> Context::outputs() { return outputs_; }

REGISTER_METHOD(mul, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  auto *out = ctx.output(0);
  out->resize(t1.shape(), t1.dtype());
  ENFORCE(t1.dtype() == t2.dtype());
  ENFORCE(t1.dtype() == Tensor::Dtype::float_);

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *d2 = static_cast<const float *>(t2.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  for (auto i = 0; i < t1.size(); ++i) {
    out_d[i] = d1[i] * d2[i];
  }
});

REGISTER_METHOD(mm, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  ENFORCE(t1.shape().size() == 2); 
  ENFORCE(t2.shape().size() == 2); 
  ENFORCE(t1.dtype() == t2.dtype());
  auto *out = ctx.output(0);
  out->resize({t1.shape()[0], t2.shape()[1]}, t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *d2 = static_cast<const float *>(t2.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
	for (auto i = 0; i < t1.shape()[0]; ++i) {
		for (auto j = 0; j < t2.shape()[1]; ++j) {
			float total = 0;
			for (auto k = 0; k < t1.shape()[1]; ++k) {
				total += d1[i * t1.shape()[1] + k] * d2[k * t2.shape()[1] + j];
			}
			out_d[i * t1.shape()[1] + j] = total;
		}
	}
});
