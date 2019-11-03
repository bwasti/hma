#include "method.h"
#include "operators/cuda/cudnn.h"

// TLS when multithreaded
static cublasHandle_t cublasHandle;
static bool cublas_initializaed;

void row_major_gemm(const Tensor& A, bool transa, const Tensor& B, bool transb, Tensor* C) {
	int m = transb ? B.shape()[0] : B.shape()[1];
	int k = transb ? B.shape()[1] : B.shape()[0];
	int k_ = transa ? A.shape()[0] : A.shape()[1];
	int n = transa ? A.shape()[1] : A.shape()[0];
	if (!(k == k_)) {
		std::stringstream ss;
		ss << "cannot multiply " << A.shape()[0] << "x" << A.shape()[1]
			 << (transa ? "^T" : "")
			 << " by " << B.shape()[0] << "x" << B.shape()[1]
			 << (transb ? "^T" : "");
		HMA_ENFORCE(k == k_, ss.str());
	}
	C->resize({n, m}, A.dtype());

	if (!cublas_initializaed) {
		checkCUBLAS(cublasCreate(&cublasHandle));
		cublas_initializaed = true;
	}

  auto transa_ = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transb_ = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
	float alpha = 1.0f, beta = 0.0f;

  checkCUBLAS(cublasSgemm(cublasHandle, transb_, transa_,
        m, n, k,
        &alpha,
        static_cast<const float*>(B.ptr()), B.shape()[1],
        static_cast<const float*>(A.ptr()), A.shape()[1],
        &beta,
        static_cast<float*>(C->ptr()), C->shape()[1]));
}

REGISTER_METHOD(CUDA, mm_nt, [](Context &ctx) {
  const auto &A = ctx.input(0);
  const auto &B = ctx.input(1);
  auto *out = ctx.output(0);
  row_major_gemm(A, false, B, true, out); return;
});

REGISTER_METHOD(CUDA, mm_nn, [](Context &ctx) {
  const auto &A = ctx.input(0);
  const auto &B = ctx.input(1);
  auto *out = ctx.output(0);
  row_major_gemm(A, false, B, false, out); return;
});

REGISTER_METHOD(CUDA, mm_tn, [](Context &ctx) {
  const auto &A = ctx.input(0);
  const auto &B = ctx.input(1);
  auto *out = ctx.output(0);
  row_major_gemm(A, true, B, false, out); return;
});

REGISTER_METHOD(CUDA, mm_tt, [](Context &ctx) {
  const auto &A = ctx.input(0);
  const auto &B = ctx.input(1);
  auto *out = ctx.output(0);
  row_major_gemm(A, true, B, true, out); return;
});

// TODO None of these are correct
REGISTER_SHAPE(mm_nt,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
							 auto co = inputs[0]->shape[0];
							 auto ci = inputs[0]->shape[1];
							 // TODO make shapes comparable
							 //HMA_ENFORCE(ci, inputs[1]->shape[0]);
							 auto n = inputs[1]->shape[1];
							 return { { n, co} };
               });

REGISTER_SHAPE(mm_nn,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
							 auto co = inputs[0]->shape[0];
							 auto ci = inputs[0]->shape[1];
							 // TODO make shapes comparable
							 //HMA_ENFORCE(ci, inputs[1]->shape[0]);
							 auto n = inputs[1]->shape[1];
							 return { { n, co} };
               });

REGISTER_SHAPE(mm_tn,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
							 auto co = inputs[0]->shape[0];
							 auto ci = inputs[0]->shape[1];
							 // TODO make shapes comparable
							 //HMA_ENFORCE(ci, inputs[1]->shape[0]);
							 auto n = inputs[1]->shape[1];
							 return { { n, co} };
               });

REGISTER_SHAPE(mm_tt,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
							 auto co = inputs[0]->shape[0];
							 auto ci = inputs[0]->shape[1];
							 // TODO make shapes comparable
							 //HMA_ENFORCE(ci, inputs[1]->shape[0]);
							 auto n = inputs[1]->shape[1];
							 return { { n, co} };
               });

REGISTER_GRAD(mm_nt, 
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                return {
									call("mm_nn", {ginputs[0], inputs[1]})[0],
									call("mm_tn", {ginputs[0], inputs[0]})[0]
									};
              });
