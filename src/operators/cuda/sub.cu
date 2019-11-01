__global__
void sub_kernel(int n, const float *x, const float *y, float *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[i] - y[i];
}

void sub(int n, const float *x, const float *y, float *z) {
  sub_kernel<<<(n+255)/256, 256>>>(n, x, y, z);
}

