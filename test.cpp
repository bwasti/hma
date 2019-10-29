#include "method.h"
#include "variable.h"
#include <ATen/ATen.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

#define SIZE 32
#define ITERS 10000

void bench_hma() {
  Graph g;
  Variable* v = g.create_var();
  v->tensor = new Tensor();
  v->tensor->resize(std::vector<size_t>{SIZE}, Tensor::Dtype::float_);

  Variable* v2 = g.create_var();
  v2->graph = &g;
  v2->tensor = new Tensor();
  v2->tensor->resize(std::vector<size_t>{SIZE}, Tensor::Dtype::float_);

  for (auto i = 0; i < v->tensor->size(); ++i) {
    ((float*)v->tensor->ptr())[i] = i;
    ((float*)v2->tensor->ptr())[i] = i * 2;
  }
  for (auto i = 0; i < ITERS; ++i) {
    v = call("add", {v, v2})[0];
  }
  auto t = resolve(v);
  assert(((float*)t->ptr())[12]);
}

void bench_torch_req_grad() {
  // I'm not entirely sure this actually enables autograd...
  at::Tensor v =  at::empty({SIZE}, at::requires_grad());
  at::Tensor v2 = at::empty({SIZE}, at::requires_grad());
  for (auto i = 0; i < v.numel(); ++i) {
    ((float*)v.data_ptr())[i] = i;
    ((float*)v2.data_ptr())[i] = i * 2;
  }
  for (auto i = 0; i < ITERS; ++i) {
    v = v + v2;
  }
  assert(((float*)v.data_ptr())[12]);
}

void bench_torch() {
  at::Tensor v = at::empty({SIZE});
  at::Tensor v2 = at::empty({SIZE});
  for (auto i = 0; i < v.numel(); ++i) {
    ((float*)v.data_ptr())[i] = i;
    ((float*)v2.data_ptr())[i] = i * 2;
  }
  for (auto i = 0; i < ITERS; ++i) {
    v = v + v2;
  }
  assert(((float*)v.data_ptr())[12]);
}

struct Bench {
  Bench(std::string tag) : tag_(tag) {
    start_ = std::chrono::steady_clock::now();
  }
  ~Bench() {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start_;
    std::cout << tag_ << ": " << diff.count() << "\n";
  }
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::string tag_;
};

int main() {
  bench_torch();
  {
    Bench _("torch\t\t");
    bench_torch();
  }
  bench_torch_req_grad();
  {
    Bench _("torch grad\t");
    bench_torch_req_grad();
  }
  bench_hma();
  {
    Bench _("hma\t\t");
    bench_hma();
  }
  return 0;
}
