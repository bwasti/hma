__global__
void neg_kernel(int n, const float *x, float *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = -x[i];
}

void neg(int n, const float *x, float *z) {
  neg_kernel<<<(n+255)/256, 256>>>(n, x, z);
}
