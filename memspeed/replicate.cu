#include<stdio.h>

__global__ void replicate(int *__restrict__ in, int *__restrict__ out, size_t n, size_t rep) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int gsize = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += gsize) {
    for (size_t j = 0; j < rep; j++) {
      out[i + j*n] = in[i];
    }
  }
}

int main() {
  int *a, *gpu_a, *gpu_b;
  int rep = 10000;
  int n = 1000;
  scanf("%d %d", &n, &rep);
  a = new int[n];
  for (int i = 0; i < n; i++) a[i] = i * 100;
  cudaMalloc(&gpu_a, sizeof(int) * n);
  cudaMalloc(&gpu_b, sizeof(int) * n * rep);
  cudaMemcpy(gpu_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);
  // seems that copying >500MB memory in one kernel is slow
  // so I divide the kernel here
  size_t div = (1<<27) / n;
  for (size_t i = 0; i < rep; i += div) {
    replicate<<<40, 256>>>(gpu_a, gpu_b + i * n, n, rep-i>div ? div : rep-i);
  }
  cudaDeviceSynchronize();
  //replicate<<<40, 256>>>(gpu_a, gpu_b, n, rep);
  cudaDeviceSynchronize();
}
