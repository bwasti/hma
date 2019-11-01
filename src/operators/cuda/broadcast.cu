__global__
void broadcast_kernel(int n, const float* x, float *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[0];
}

void broadcast(int n, const float* x, float *z) {
  broadcast_kernel<<<(n+255)/256, 256>>>(n, x, z);
}

