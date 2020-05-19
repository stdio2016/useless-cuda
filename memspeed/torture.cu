#include<stdio.h>
__global__ void pattern(int *__restrict a, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    a[i] = (long long)i*77%n;
  } 
}
__global__ void torture(int *__restrict a, int *__restrict b, int *__restrict c, int n) {
  int s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  int i = threadIdx.x;
  int gs = blockDim.x * gridDim.x;
  int b1=0,b2=0,b3=0,b4=0;
  for (i = i; i < n-gs*3; i += gs * 4) {
    int aa1 = a[i], aa2 = a[i+gs], aa3 = a[i+gs*2], aa4 = a[i+gs*3];
    s1 += b1;
    b1 = b[aa1];
    s2 += b2;
    b2 = b[aa2];
    s3 += b3;
    b3 = b[aa3];
    s4 += b4;
    b4 = b[aa4];
  }
  s1 += b1; s2+=b2; s3+=b3; s4+=b4;
  c[threadIdx.x + blockIdx.x * blockDim.x] = s1 + s2 + s3 + s4;
}
__global__ void torture2(int *__restrict a, int *__restrict b, int *__restrict c, int n) {
  int s = 0;
  int i = threadIdx.x;
  int gs = blockDim.x * gridDim.x;
  for (i = i; i < n; i += gs) {
    s += b[a[i]];
  }
  c[threadIdx.x + blockIdx.x * blockDim.x] = s;
}
int main(){
  int *a,*b,*c, d;
  int n = 200000000;
  cudaMalloc(&a, sizeof(int) * n);
  cudaMalloc(&b, sizeof(int) * n);
  cudaMalloc(&c, sizeof(int) * 10000);
  pattern<<<10,1024>>>(a, n);
  cudaDeviceSynchronize();
  for (int i = 0; i < 10; i++)
  torture2<<<10,1024>>>(a, b, c, n);
  cudaMemcpy(&d, c, sizeof(int), cudaMemcpyDeviceToHost);
}