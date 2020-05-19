#include<stdio.h>
#include<cstdlib>

__global__ void race(volatile int *arr, size_t size, int n, int *out) {
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  int ptr = gid;
  if (ptr > size) ptr = 0;
  for (int i = 0; i < n; i++) {
    ptr = arr[size * gid + ptr];
  }
  out[gid] = ptr;
}

int main(int argc, char *argv[]) {
  int grid_size=10, block_size=128, n=100, step=100;
  if (argc > 4) {
    sscanf(argv[1], "%d", &grid_size);
    sscanf(argv[2], "%d", &block_size);
    sscanf(argv[3], "%d", &n);
    sscanf(argv[4], "%d", &step);
  }
  size_t size = n;
  size_t total_size = size * grid_size * block_size * sizeof(int);
  printf("size = %zd KB\n", total_size / 1024);
  int *arr = new int[size * grid_size * block_size];
  int *ra = new int[size];
  int *out = new int[grid_size * block_size];
  for (size_t tid = 0; tid < grid_size * block_size; tid++) {
    int *arr2 = arr + tid * size;
    for (int i = 0; i < n; i++) {
      ra[i] = i;
    }
    for (int i = 1; i < n; i++) {
      int r = rand()%(i+1);
      int tmp = ra[i];
      ra[i] = ra[r];
      ra[r] = tmp;
    }
    /*for (int i = 1; i < n; i++) {
      arr2[ra[i-1]] = ra[i];
    }
    arr2[ra[n-1]] = ra[0];*/
    for (int i = 1; i < n; i++) {
      arr2[i-1] = i;
    }
    arr2[n-1] = 0;
  }
  int *garr, *gout;
  cudaMalloc(&garr, total_size);
  cudaMalloc(&gout, sizeof(int) * grid_size * block_size);
  cudaMemcpy(garr, arr, total_size, cudaMemcpyHostToDevice);
  race<<<grid_size, block_size>>>(garr, size, step, gout);
  cudaMemcpy(out, gout, sizeof(int) * grid_size * block_size, cudaMemcpyDeviceToHost);
  return 0;
}