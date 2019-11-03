#pragma once

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define checkCUDNN(expression) {                             \
	cudnnStatus_t status = (expression);                     \
	HMA_ENFORCE(status == CUDNN_STATUS_SUCCESS,              \
			std::string(cudnnGetErrorString(status)));   \
}

#define checkCuda(status) {                            \
	std::stringstream _error;                                          \
	HMA_ENFORCE(status == CUDA_SUCCESS,              \
			std::string(cudaGetErrorString(status)));   \
}

#define checkCUBLAS(status) {                            \
	HMA_ENFORCE(status == CUBLAS_STATUS_SUCCESS);              \
}
