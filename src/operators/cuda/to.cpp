#include "alloc.h"
#include "method.h"
#include <cuda_runtime.h>

void *cuda_malloc(size_t size) {
  void *out;
  // TODO check error
  auto res = cudaMalloc(&out, size);
  HMA_ENFORCE(res == cudaSuccess);
  return out;
}

void cuda_free(void *ptr) {
  // TODO check error
  auto res = cudaFree(ptr);
  HMA_ENFORCE(res == cudaSuccess);
}

REGISTER_ALLOC(CUDA, &cuda_malloc, &cuda_free);

REGISTER_METHOD(CPU, to_cuda, [](Context &ctx) {
  const auto &t = ctx.input(0);
  auto *out = ctx.output(0);
  out->setTag(getTag("CUDA"));
  out->resize(t.shape(), t.dtype());

  auto *d = static_cast<const float *>(t.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  auto res = cudaMemcpy(out_d, d, t.bytes(), cudaMemcpyHostToDevice);
  HMA_ENFORCE(res == cudaSuccess);
});

REGISTER_METHOD(CUDA, to_cpu, [](Context &ctx) {
  const auto &t = ctx.input(0);
  auto *out = ctx.output(0);
  out->setTag(getTag("CPU"));
  out->resize(t.shape(), t.dtype());

  auto *d = static_cast<const float *>(t.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  auto res = cudaMemcpy(out_d, d, t.bytes(), cudaMemcpyDeviceToHost);
  HMA_ENFORCE(res == cudaSuccess);
});
