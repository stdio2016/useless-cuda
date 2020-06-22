/**
 * Inspired from
 * https://forums.developer.nvidia.com/t/n-queen-solver-for-cuda/5369/15
 * https://forum.beyond3d.com/threads/n-queen-solver-for-opencl.47785/
 * http://masm32.com/board/index.php?topic=7848.0
 * Optimization of N-Queens Solvers on Graphics Processors. Tao Zhang, Wei Shu,
   Min-You Wu. In: Proceedings of the 9th International Conference on Advanced
   Parallel Processing Technologies 142-156
 */
#include <stdio.h>
#include <cstring>
#include <vector>

typedef unsigned uint;

struct SubP {
  uint mid, diag1, diag2;
  uint ss;
};

int gpu_workCount;
SubP *gpu_works;
int *gpu_flag;
long long *gpu_result;
uint *gpu_canplace;

__global__ void nqueen_kern(int lv, uint canplace[], struct SubP *works, int workCount,
        int *flag, long long *result) {
  __shared__ uint s0[12][256];
  int tid = threadIdx.x;
  int bid = blockIdx.x, bsize = blockDim.x;
  if (bsize != 256) return;
  int gid = tid + bid * bsize;
  if (gid >= workCount) {
    result[gid] = 0;
    return;
  }

  int i = lv-1;
  long long sol = 0;
  int my = gid;
  uint mid = works[my].mid;
  uint diag1 = works[my].diag1, diag1r = 0;
  uint diag2 = works[my].diag2, diag2r = 0;
  s0[i][tid] = canplace[i] & ~(mid | diag1 | diag2);
  for (; my < workCount; ) {
    uint can = s0[i][tid];
    uint s = can & -can;
    if (can) {
      mid += s;
      diag1 += s;
      diag1r = diag1r << 1 | diag1 >> 31;
      diag1 <<= 1;
      diag2 += s;
      diag2r = diag2r >> 1 | diag2 << 31;
      diag2 >>= 1;
      i -= 1;
      can = canplace[i] & ~(mid | diag1 | diag2);
      s0[i][tid] = can;
    }
    if (can && i == 0) sol += 1, can = 0;
    if (!can && i+1<lv) {
      i += 1;
      can = s0[i][tid];
      s = can & -can;
      mid -= s;
      diag1 = diag1r << 31 | diag1 >> 1;
      diag1r >>= 1;
      diag1 -= s;
      diag2 = diag2r >> 31 | diag2 << 1;
      diag2r <<= 1;
      diag2 -= s;
      s0[i][tid] = can - s;
    }
    if (!can && i == lv-1) {
      my = atomicAdd(flag, 1);
      if (my < workCount) {
        mid = works[my].mid;
        diag1 = works[my].diag1, diag1r = 0;
        diag2 = works[my].diag2, diag2r = 0;
        s0[i][tid] = canplace[i] & ~(mid | diag1 | diag2);
      }
    }
  }
  result[gid] = sol;
}

void nqueen_send_gpu(int cut, int nblocks, int nworks, const SubP *works) {
  int flag = nblocks * 256;
  cudaMemcpy(gpu_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_works, works, nworks * sizeof(SubP), cudaMemcpyHostToDevice);
  nqueen_kern<<<nblocks, 256>>>(cut, gpu_canplace, gpu_works, nworks,
    gpu_flag, gpu_result);
}

long long nqueen_wait_compute(int nblocks) {
  long long *result = new long long[nblocks*256];
  long long sum = 0;

  cudaDeviceSynchronize();
  cudaMemcpy(result, gpu_result, nblocks*256 * sizeof(long long), cudaMemcpyDeviceToHost);

  for (int i = 0; i < nblocks*256; i++) {
    sum += result[i];
  }

  delete[] result;
  return sum;
}

long long nqueen_gen(int lv, int cut, uint mid, uint diag1, uint diag2
        , uint *canplace, std::vector<SubP> &works, int nblocks, int nworks) {
  uint s0[32], s1[32], s2[32], s3[32];
  if (lv < cut) return 0;
  int i = lv-1;
  uint choice = canplace[lv-1] & ~(mid | diag1 | diag2);
  s0[i] = choice;
  s1[i] = mid;
  s2[i] = diag1;
  s3[i] = diag2;
  long long sum = 0;
  int lastWork = 0;
  while (i < lv) {
    choice = s0[i];
    mid = s1[i];
    diag1 = s2[i];
    diag2 = s3[i];
    uint bit = choice & -choice;
    s0[i] = choice - bit;
    if (!choice) i++;
    else if (i == cut-1) {
      SubP p = {mid, diag1, diag2};
      works.push_back(p);
      if (works.size() >= nworks) {
        if (lastWork) {
          sum += nqueen_wait_compute(nblocks);
        }
        nqueen_send_gpu(cut, nblocks, nworks, works.data());
        lastWork = nworks;
        works.clear();
      }
      i++;
    }
    else {
      mid = mid + bit;
      diag1 = (diag1 | bit) << 1;
      diag2 = (diag2 | bit) >> 1;
      i -= 1;
      choice = canplace[i] & ~(mid | diag1 | diag2);
      s0[i] = choice;
      s1[i] = mid;
      s2[i] = diag1;
      s3[i] = diag2;
      if (choice == 0) i += 1;
    }
  }
  if (lastWork) {
    sum += nqueen_wait_compute(nblocks);
  }
  nqueen_send_gpu(cut, nblocks, works.size(), works.data());
  sum += nqueen_wait_compute(nblocks);
  return sum;
}

long long nqueen_cuda(int n, unsigned canplace[], int nblocks, int nworks) {
  std::vector<SubP> works;
  works.reserve(10000*32);
  // create subproblems
  int cut = n - 5;
  if (cut < 2) cut = 2;
  if (cut > 12) cut = 12;
  cudaMemcpy(gpu_canplace, canplace, sizeof(uint) * 32, cudaMemcpyHostToDevice);
  return nqueen_gen(n, cut, 0, 0, 0, canplace, works, nblocks, nworks);
}

int main(int argc, char *argv[]) {
  int n;
  int T = 0;
  char buf[100];
  unsigned canplace[32];
  FILE *filein = stdin;
  for (int i = 1; i < argc; i++) {
    if (i+1<argc && filein == stdin && strcmp(argv[i], "-i") == 0) {
      filein = fopen(argv[i+1], "r");
      if (filein == NULL) {
        fprintf(stderr, "cannot open file\n");
        return 1;
      }
    }
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int nblocks = prop.multiProcessorCount * 4;
  gpu_workCount = nblocks * 256 * 30;
  cudaMalloc(&gpu_works, sizeof(SubP) * gpu_workCount);
  cudaMalloc(&gpu_flag, sizeof(int));
  cudaMalloc(&gpu_canplace, sizeof(uint) * 32);
  cudaMalloc(&gpu_result, sizeof(long long) * nblocks * 256);

  while (fscanf(filein, "%d", &n) == 1) {
    fgets(buf, 100, filein);
    T += 1;
    if (n < 1 || n >= 32) return 0;
    for (int i = 0; i < n; i++) {
      fgets(buf, 100, filein);
      canplace[n-1-i] = (1u<<n)-1;
      for (int j = 0; j < n; j++) {
        if (buf[j] == '*') canplace[n-1-i] -= 1u<<j;
      }
    }

    long long ans = 0;
    ans = nqueen_cuda(n, canplace, nblocks, gpu_workCount);
    printf("Case #%d: %lld\n", T, ans);
  }
  return 0;
}
