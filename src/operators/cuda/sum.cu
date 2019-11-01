__global__
void sum_kernel(int n, const float *x, float *z) {
	extern __shared__ float sdata[];
	int offset = threadIdx.x * 256;
  float total = 0;
  for (int i = 0; i < 256; ++i) {
		if (offset + i < n) {
			total += x[offset + i];
		}
	}
	sdata[threadIdx.x] = total;
	__syncthreads();
  if (offset == 0) {
    float ttotal = 0;
		for (int i = 0; i < 256; ++i) {
			ttotal += sdata[i];
		}
		z[0] = ttotal;
	}
}

void sum(int n, const float *x, float *z) {
	sum_kernel<<<1, 256, 256 * sizeof(float)>>>(n, x, z);
}

