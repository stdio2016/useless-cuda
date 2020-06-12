#include<stdio.h>
#include<cstdint>
#include<thrust/device_ptr.h>
#include<thrust/scan.h>

typedef uint32_t u32;


__global__ void count_gen(u32 *src, int nsrc, u32 *choice, int nchoices, int *ngen) {
  __shared__ u32 some[256];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  for (int i = threadIdx.x; i < nchoices; i += blockDim.x) {
    some[i] = choice[i];
  }
  __syncthreads();
  for (int i = tid; i < nsrc; i += nthreads) {
    int sum = 0;
    u32 a = src[i];
    for (int j = 0; j < nchoices; j++) {
      if ((a & some[j]) == 0) {
        sum += 1;
      }
    }
    ngen[i] = sum;
  }
}

__global__ void gen(u32 *src, int nsrc, u32 *choice, int nchoices, u32 *dest, int *ngen) {
  __shared__ u32 some[256];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nthreads = blockDim.x * gridDim.x;
  for (int i = threadIdx.x; i < nchoices; i += blockDim.x) {
    some[i] = choice[i];
  }
  __syncthreads();
  for (int i = tid; i < nsrc; i += nthreads) {
    u32 a = src[i];
    int pos = ngen[i];
    for (int j = 0; j < nchoices; j++) {
      u32 added = a | some[j];
      int canplace = (a & some[j]) == 0;
      if (canplace) {
        dest[pos] = added;
      }
      pos += canplace;
    }
  }
}

int main() {
  int nsrc = 1000000, nchoices = 100, bufsize = 30000000;
  u32 *gsrc, *gchoice, *gdest;
  int *gcnt;
  cudaMalloc(&gsrc, sizeof(u32) * nsrc);
  cudaMalloc(&gchoice, sizeof(u32) * nchoices);
  cudaMalloc(&gdest, sizeof(u32) * bufsize);
  cudaMalloc(&gcnt, sizeof(int) * (nsrc+1));
  u32 *src, *choice, *dest;
  src = new u32[nsrc];
  choice = new u32[nchoices];
  dest = new u32[bufsize];
  unsigned seed = 2;
  for (int i = 0; i < nsrc; i++) {
    src[i] = seed;
    seed = seed*0xdefaced + 1;
  }
  for (int i = 0; i < nchoices; i++) {
    u32 rr = seed;
    seed = seed*0xdefaced + 1;
    choice[i] = 0;
    for (int yy = 0; yy < 5; yy++) {
      choice[i] += 1<<(rr>>yy*5&31);
    }
  }
  cudaMemcpy(gsrc, src, sizeof(u32) * nsrc, cudaMemcpyHostToDevice);
  cudaMemcpy(gchoice, choice, sizeof(u32) * nchoices, cudaMemcpyHostToDevice);
  cudaMemset(gcnt, 0, sizeof(int) * (nsrc+1));
  int ans = 0;
  count_gen<<<40, 256>>>(gsrc, nsrc, gchoice, nchoices, gcnt);
  thrust::device_ptr<int> thcnt = thrust::device_pointer_cast(gcnt);
  thrust::exclusive_scan(thcnt, thcnt + nsrc+1, thcnt);
  cudaMemcpy(&ans, gcnt+nsrc, sizeof(int), cudaMemcpyDeviceToHost);
  gen<<<40, 256>>>(gsrc, nsrc, gchoice, nchoices, gdest, gcnt);
  cudaMemcpy(dest, gdest, sizeof(u32) * ans, cudaMemcpyDeviceToHost);
  printf("generated: %d\n", ans);
}
