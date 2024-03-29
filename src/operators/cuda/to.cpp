#include "alloc.h"
#include "method.h"
#include "operators/tag_like.h"
#include <cuda_runtime.h>

void *cuda_malloc(size_t size) {
  void *out;
  auto res = cudaMalloc(&out, size);
  HMA_ENFORCE(res == cudaSuccess);
  return out;
}

void cuda_free(void *ptr) {
  auto res = cudaFree(ptr);
  HMA_ENFORCE(res == cudaSuccess);
}

void* cuda_memcpy(void *dst, const void* src, size_t bytes) {
  auto res = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
  HMA_ENFORCE(res == cudaSuccess);
  return dst;
}

REGISTER_ALLOC(CUDA, &cuda_malloc, &cuda_free, &cuda_memcpy);

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

REGISTER_TAG_PAIR(CUDA, CPU, to_cpu);
REGISTER_TAG_PAIR(CPU, CUDA, to_cuda);
REGISTER_METHOD(CUDA, tag_like, genericTagLike);

REGISTER_SHAPE(to_cpu,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
                 return {inputs[0]->shape};
               });

REGISTER_SHAPE(to_cuda,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
                 return {inputs[0]->shape};
               });
