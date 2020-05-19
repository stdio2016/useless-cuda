#include<stdio.h>
__global__ void pattern(int *__restrict a, int n, int m) {
  for (int i = threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    a[i] = (long long)i*77%m;
  } 
}
__global__ void torture2(int *__restrict a, int *__restrict b, int *__restrict c, int n) {
  int s = 0;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int gs = blockDim.x * gridDim.x;
  for (i = i; i < n; i += gs) {
    int aa = a[i];
    s += b[aa];
    s += b[aa+1];
    /*s += b[aa+2];
    s += b[aa+3];*/
  }
  c[threadIdx.x + blockIdx.x * blockDim.x] = s;
}
int main(){
  int *a,*b,*c, d;
  int n = 100000, m = 100000;
  cudaMalloc(&a, sizeof(int) * n);
  cudaMalloc(&b, sizeof(int) * m);
  cudaMalloc(&c, sizeof(int) * 10000);
  pattern<<<10,1024>>>(a, n, m);
  cudaDeviceSynchronize();
  for (int i = 0; i < 3000; i++)
  torture2<<<10,1024>>>(a, b, c, n);
  cudaMemcpy(&d, c, sizeof(int), cudaMemcpyDeviceToHost);
}