#include "method.h"
#include "operators/cuda/cudnn.h"

void setConvParams(
	cudnnTensorDescriptor_t& srcTensorDesc,
	cudnnFilterDescriptor_t& filterDesc,
	cudnnTensorDescriptor_t& dstTensorDesc,
	cudnnConvolutionDescriptor_t& convDesc,
	int& n,
	int& c,
	int& h,
	int& w,
	int& m,
  int& k) {
	checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				n, c,
				h, w));

	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
				CUDNN_DATA_FLOAT,
				CUDNN_TENSOR_NCHW,
				m,
				c,
				k,
				k));

	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
				0, 0,
				1, 1,
				1, 1,
				CUDNN_CROSS_CORRELATION,
				CUDNN_DATA_FLOAT));

	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
				srcTensorDesc,
				filterDesc,
				&n, &c, &h, &w));

	checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				n, c,
				h, w));
}

REGISTER_METHOD(CUDA, conv, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  // bias
  if (ctx.num_inputs() > 2) {
    HMA_ENFORCE(0, std::string("bias not yet supported"));
		const auto &t3 = ctx.input(2);
	}
  auto *out = ctx.output(0);

	int n = t1.shape()[0];
	int c = t1.shape()[1];
	int h = t1.shape()[2];
	int w = t1.shape()[3];

	int m = t2.shape()[0];
	int c_ = t2.shape()[1];
  HMA_ENFORCE(c == c_);
  int k = t2.shape()[1];
  int k_ = t2.shape()[2];
  HMA_ENFORCE(k == k_);

	cudnnHandle_t cudnnHandle;
	checkCUDNN(cudnnCreate(&cudnnHandle));

	cudnnTensorDescriptor_t srcTensorDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnTensorDescriptor_t dstTensorDesc;

	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t convAlgo;

	checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	setConvParams(srcTensorDesc, filterDesc, dstTensorDesc, convDesc,
			n, c, h, w, m, k);

  out->resize({n, c, h, w}, t1.dtype());

	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
				srcTensorDesc,
				filterDesc,
				convDesc,
				dstTensorDesc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				0,
				&convAlgo));

	size_t sizeInBytes = 0;

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
				srcTensorDesc,
				filterDesc,
				convDesc,
				dstTensorDesc,
				convAlgo,
				&sizeInBytes));

	Tensor workspace(getTag("CUDA"));
	workspace.resize({sizeInBytes}, Tensor::Dtype::byte_);

	float alpha = 1.0f, beta = 0.0f;

	checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, srcTensorDesc,
				t1.ptr(), filterDesc, t2.ptr(), convDesc, 
				convAlgo, workspace.ptr(), sizeInBytes, &beta,
				dstTensorDesc, out->ptr()));
});

REGISTER_METHOD(CUDA, conv_dx, [](Context &ctx) {
  // input
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  // output
	const auto &t3 = ctx.input(2);
  auto *out = ctx.output(0);
	out->resize(t1.shape(), t1.dtype());

	int n = t1.shape()[0];
	int c = t1.shape()[1];
	int h = t1.shape()[2];
	int w = t1.shape()[3];

	int m = t2.shape()[0];
	int c_ = t2.shape()[1];
  HMA_ENFORCE(c == c_);
  int k = t2.shape()[1];
  int k_ = t2.shape()[2];
  HMA_ENFORCE(k == k_);

	cudnnHandle_t cudnnHandle;
	checkCUDNN(cudnnCreate(&cudnnHandle));

	cudnnTensorDescriptor_t srcTensorDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnTensorDescriptor_t dstTensorDesc;

	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionBwdDataAlgo_t dalgo;

	checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));

	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	setConvParams(srcTensorDesc, filterDesc, dstTensorDesc, convDesc,
			n, c, h, w, m, k);

	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
				cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &dalgo));

	size_t tmpsize = 0;

	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
				cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc, 
				dalgo, &tmpsize));

	Tensor workspace(getTag("CUDA"));
	workspace.resize({tmpsize}, Tensor::Dtype::byte_);

	float alpha = 1.0f, beta = 0.0f;

	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, 
				filterDesc, t2.ptr(), dstTensorDesc, t3.ptr(),
				convDesc, dalgo, workspace.ptr(), tmpsize, &beta, srcTensorDesc,
				out->ptr()));

});

REGISTER_METHOD(CUDA, conv_dw, [](Context &ctx) {
  // input
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  // grad input
  const auto &t3 = ctx.input(2);
  auto *out = ctx.output(0);
	out->resize(t2.shape(), t2.dtype());

	int n = t1.shape()[0];
	int c = t1.shape()[1];
	int h = t1.shape()[2];
	int w = t1.shape()[3];

	int m = t2.shape()[0];
	int c_ = t2.shape()[1];
  HMA_ENFORCE(c == c_);
  int k = t2.shape()[1];
  int k_ = t2.shape()[2];
  HMA_ENFORCE(k == k_);

	cudnnHandle_t cudnnHandle;
	checkCUDNN(cudnnCreate(&cudnnHandle));

	cudnnTensorDescriptor_t srcTensorDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnTensorDescriptor_t dstTensorDesc;

	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionBwdFilterAlgo_t dalgo;

	checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));

	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	setConvParams(srcTensorDesc, filterDesc, dstTensorDesc, convDesc,
			n, c, h, w, m, k);

	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
				cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
				CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &dalgo));

	size_t tmpsize = 0;

	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
				cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, 
				dalgo, &tmpsize));

	Tensor workspace(getTag("CUDA"));
	workspace.resize({tmpsize}, Tensor::Dtype::byte_);

	float alpha = 1.0f, beta = 0.0f;

	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, 
				srcTensorDesc, t1.ptr(), dstTensorDesc, t3.ptr(),
				convDesc, dalgo, workspace.ptr(), tmpsize, &beta, filterDesc,
				out->ptr()));

});

REGISTER_GRAD(conv,
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                return {call("conv_dx", {inputs[0], inputs[1], ginputs[0]})[0],
												call("conv_dw", {inputs[0], inputs[1], ginputs[0]})[0]
												};
              });


REGISTER_SHAPE(conv,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
							 auto h = inputs[0]->shape[2];
							 auto w = inputs[0]->shape[3];
							 return { 
							 	{ inputs[0]->shape[0], inputs[1]->shape[0], h, w }
							 };
               });

REGISTER_SHAPE(conv_dx,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
							 return { 
							 	{ inputs[0]->shape }
							 };
               });

REGISTER_SHAPE(conv_dw,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
							 return { 
							 	{ inputs[0]->shape }
							 };
               });
