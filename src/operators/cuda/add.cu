__global__
void add_kernel(int n, const float *x, const float *y, float *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[i] + y[i];
}

void add(int n, const float *x, const float *y, float *z) {
  add_kernel<<<(n+255)/256, 256>>>(n, x, y, z);
}
