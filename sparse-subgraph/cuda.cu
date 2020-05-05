#include<iostream>

typedef uint64_t u64;

/* kernel input:
 *   E: adjacency matrix as bit mask, (E[i]>>j & 1) is E(i,j)
 *   n: number of vertices. must be 5 ~ 64
 *   ans: output answer, ans[i*64+j] is j+1 sparsest subgraph from i-th block
 *   lv: tweak the kernel. must be 1 ~ n-4 and less than 7
*/
__global__ void subgraph_kern(u64 *E, int n, int *ans, int lv) {
  const int max_lv = 7;
  int laneid = threadIdx.x & 31;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;
  int gsize = blockDim.x * gridDim.x;
  __shared__ int s[64][32];
  
  for (int i = tid>>5; i < 64; i += blockDim.x>>5) {
    s[i][laneid] = 100000;
  }
  __syncthreads();

  int good = 0;
  int bu[16] = {0};
  for (int i = 1; i < 16; i++) {
    bu[i] = __popc(E[31-__clz(i&-i)] & (i ^ i>>1));
  }
  for (u64 t = gid; t < 1ull<<(n-(lv+4)); t += gsize) {
    int s0, s1, s2, s3, s4;
    s0 = s1 = s2 = s3 = s4 = 100000;
    // shift register, to reduce shared memory usage
    int sL[max_lv], sR[max_lv];
    for (int j = 0; j < max_lv; j++) sL[j] = 100000;
    for (int j = 0; j < max_lv; j++) sR[j] = 100000;
    // get subproblem
    u64 actual = t<<(lv+4);
    good = 0;
    for (int j = lv+4; j < n; j++) {
      if (actual>>j & 1) {
        good += __popcll(actual & E[j]);
      }
    }
    good >>= 1;

    for (int i = 0; i < 1<<lv; i += 1) {
      u64 mask = actual + ((i ^ i>>1) << 4);
      if (i) {
        int z = 31-__clz(i&-i);
        int diff = __popcll(E[z+4] & mask);
        if ((i ^ i>>1)>>z & 1) {
          // add vertex
          good += diff;
          #pragma loop unroll
          for (int j = max_lv-1; j > 0; j--) sL[j] = sL[j-1];
          sL[0] = s0;
          s0 = s1; s1 = s2; s2 = s3; s3 = s4;
          s4 = sR[0];
          #pragma loop unroll
          for (int j = 0; j < max_lv-1; j++) sR[j] = sR[j+1];
        }
        else {
          // remove vertex
          good -= diff;
          #pragma unroll
          for (int j = max_lv-1; j > 0; j--) sR[j] = sR[j-1];
          sR[0] = s4;
          s4 = s3; s3 = s2; s2 = s1; s1 = s0;
          s0 = sL[0];
          #pragma unroll
          for (int j = 0; j < max_lv-1; j++) sL[j] = sL[j+1];
        }
      }
      int g = good;

      int E0 = __popcll(E[0] & mask);
      int E1 = __popcll(E[1] & mask);
      int E2 = __popcll(E[2] & mask);
      int E3 = __popcll(E[3] & mask);
      if (g < s0) s0 = g;
      g += E0 + bu[1];
      if (g < s1) s1 = g;
      g += E1 + bu[2];
      if (g < s2) s2 = g;
      g -= E0 + bu[3];
      if (g < s1) s1 = g;

      g += E2 + bu[4];
      if (g < s2) s2 = g;
      g += E0 + bu[5];
      if (g < s3) s3 = g;
      g -= E1 + bu[6];
      if (g < s2) s2 = g;
      g -= E0 + bu[7];
      if (g < s1) s1 = g;

      g += E3 + bu[8];
      if (g < s2) s2 = g;
      g += E0 + bu[9];
      if (g < s3) s3 = g;
      g += E1 + bu[10];
      if (g < s4) s4 = g;
      g -= E0 + bu[11];
      if (g < s3) s3 = g;

      g -= E2 + bu[12];
      if (g < s2) s2 = g;
      g += E0 + bu[13];
      if (g < s3) s3 = g;
      g -= E1 + bu[14];
      if (g < s2) s2 = g;
      g -= E0 + bu[15];
      if (g < s1) s1 = g;
    }
    int b = __popcll(actual);
    if (b) atomicMin(&s[b-1][laneid], sL[0]);
    atomicMin(&s[b+0][laneid], s0);
    atomicMin(&s[b+1][laneid], s1);
    atomicMin(&s[b+2][laneid], s2);
    atomicMin(&s[b+3][laneid], s3);
    atomicMin(&s[b+4][laneid], s4);
    for (int j = 0; j < max_lv-1; j++) {
      atomicMin(&s[b+5+j][laneid], sR[j]);
    }
  }
  // combine result from each thread
  __syncthreads();
  for (int step = 16; step >= 1; step>>=1) {
    for (int i = tid>>5; i < 64; i += blockDim.x>>5) {
      if (laneid < step)
        s[i][laneid] = min(s[i][laneid], s[i][laneid+step]);
    }
    __syncthreads();
  }
  for (int i = tid; i < 64; i += blockDim.x) {
    ans[blockIdx.x * 64 + i] = s[i][0];
  }
}

int mypopcnt(unsigned x) {
  int n = 0;
  for (int i = 0; i < 32; i++) {
    n += x>>i & 1;
  }
  return n;
}

int main() {
  int n;
  u64 E[64];
  std::cin >> n;
  if (n < 0 || n > 64) exit(1);
  for(int i=0;i<n;i++){
    u64 good = 0, y = 0;
    for(int j=0;j<n;j++){
      u64 w;
      if (std::cin>>w) good = 1;
      y|=w<<j;
    }
    if (good) E[i] = y;
  }
  
  int blocksize = 50;
  int lv = n - 12;
  int *ans;
  if (lv < 1) lv = 1;
  if (lv > 7) lv = 7;
  if (n < 19) blocksize = 1;
  else if (n < 25) blocksize = 1<<(n-19);
  ans = new int[64 * blocksize];

  if (n >= 5) {
    u64 *gpu_E;
    int *gpu_ans;
    cudaMalloc(&gpu_E, sizeof(u64) * 64);
    cudaMalloc(&gpu_ans, sizeof(int) * 64 * blocksize);
    cudaMemcpy(gpu_E, E, sizeof(u64) * 64, cudaMemcpyHostToDevice);
    subgraph_kern<<<blocksize, 256>>>(gpu_E, n, gpu_ans, lv);
    cudaMemcpy(ans, gpu_ans, sizeof(int) * 64 * blocksize, cudaMemcpyDeviceToHost);
    for (int i = 1; i < blocksize; i++) {
      for (int j = 0; j < 64; j++) {
        if (ans[i*64+j] < ans[j]) ans[j] = ans[i*64+j];
      }
    }
  }
  else {
    // small case
    for (int j = 0; j < 64; j++) ans[j] = 100000;
    for (int i = 0; i < 1<<n; i++) {
      int z = mypopcnt(i);
      int sum = 0;
      for (int j = 0; j < n; j++) {
        if (i>>j & 1)
          sum += mypopcnt(i & E[j]);
      }
      sum >>= 1;
      if (z && sum < ans[z-1]) ans[z-1] = sum;
    }
  }
  for (int i = 1; i < n; i++) {
    std::cout << i+1 << ' ' << ans[i] << '\n';
  }
  return 0;
}
