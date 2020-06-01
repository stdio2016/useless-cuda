// random read but coalesced
#include<stdio.h>
#include<cstdlib>

__global__ void race(volatile int *arr, size_t size, int n, int *out) {
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  int ptr = gid;
  if (ptr > size) ptr = 0;
  ptr = arr[ptr];
  #pragma unroll 16
  for (int i = 0; i < n; i++) {
    ptr = __ballot_sync(0xffffffff, ptr);
  }
  out[gid] = ptr;
}

int main(int argc, char *argv[]) {
  int grid_size=10, block_size=128, n=100, step=100;
  int warp_size = 32;
  if (argc > 4) {
    sscanf(argv[1], "%d", &grid_size);
    sscanf(argv[2], "%d", &block_size);
    sscanf(argv[3], "%d", &n);
    sscanf(argv[4], "%d", &step);
  }

  // make sure block size is multiple of warp size
  block_size -= block_size % warp_size;
  n -= n % warp_size;

  size_t size = n;
  size_t total_size = size * sizeof(int);
  printf("size = %zd KB\n", total_size / 1024);
  int *arr = new int[size];
  int *ra = new int[size];
  int *out = new int[grid_size * block_size];
  {
    // create random permutation
    for (int i = 0; i < n/warp_size; i++) {
      ra[i] = i;
    }
    for (int i = 1; i < n/warp_size; i++) {
      int r = rand()%(i+1);
      int tmp = ra[i];
      ra[i] = ra[r];
      ra[r] = tmp;
    }
    // create "coalesced" random cycle
    for (int i = 1; i < n/warp_size; i++) {
      for (int j = 0; j < warp_size; j++) {
        arr[ra[i-1]*warp_size + j] = ra[i]*warp_size + j;
      }
    }
    for (int j = 0; j < warp_size; j++) {
      arr[ra[n/warp_size-1]*warp_size + j] = ra[0]*warp_size + j;
    }
  }
  int *garr, *gout;
  cudaMalloc(&garr, total_size);
  cudaMalloc(&gout, sizeof(int) * grid_size * block_size);
  cudaMemcpy(garr, arr, total_size, cudaMemcpyHostToDevice);
  race<<<grid_size, block_size>>>(garr, size, step, gout);
  cudaMemcpy(out, gout, sizeof(int) * grid_size * block_size, cudaMemcpyDeviceToHost);
  if (block_size * grid_size > 123) {
    printf("out[123] = %d\n", out[123]);
  }
  return 0;
}
