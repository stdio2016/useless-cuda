// counting Hamilton cycle, CUDA acceleration
#include<stdio.h>
#include<stdlib.h>
#define MAX_BLOCK_SIZE 256
#define MAX_ARRAY_SIZE (1024*8)
typedef unsigned long long u64;

// any 2 <= mod <= 2^31 should work
__host__ __device__ unsigned mod_sum(unsigned a, unsigned b, unsigned mod) {
    unsigned c = a+b;
    return c >= mod ? c-mod : c;
}

__host__ __device__ u64 mod_sum64(u64 a, u64 b, u64 mod) {
    u64 c = a+b;
    return c >= mod ? c-mod : c;
}

template<int k>
__launch_bounds__(MAX_BLOCK_SIZE)
__global__ void ha2(int n, int work, unsigned *part, int *adj, unsigned long long *ret, unsigned long long mod) {
    __shared__ unsigned long long qc[1024]; // transition count
    __shared__ unsigned long long ai[64]; // adjacency matrix as bitset
    //const int k = blockDim.x;
    const int tid = threadIdx.x;
    const int bid = threadIdx.y + blockIdx.x * blockDim.y;
    const int sha = threadIdx.y * k;
    const int gridSize = blockDim.y * gridDim.x;
    unsigned long long s = part[bid];
    unsigned long long mask = (1ull<<k) - 1;
    unsigned long long total = 0;

    // fetch adjacency matrix
    for (int i = tid+sha; i < n; i += blockDim.y * k) {
        unsigned long long aa = 0;
        for (int j = 0; j < n; j++) {
            aa = aa | static_cast<unsigned long long>(adj[i * n + j]) << j;
        }
        ai[i] = aa;
    }
    __syncthreads();

    for (int runs = 0; runs < work; runs += gridSize) {
        unsigned at;
        {
            unsigned long long row = s;
            for (int i = 0; i < tid; i++) {
                row = row & (row-1);
            }
            at = __ffsll(row)-1;
        }
        // making row "long long" would make program 3x slow, so I use 2 unsigned int
        unsigned row = 0, row2 = 0;
        {
            // build transition table
            unsigned long long me = ai[at];
            for (int i = n-2; i >= 0; i--) {
                if (s>>i & 1) {
                    row2 = row2 << 1 | row >> 31;
                    row = row + row + (me>>i & 1);
                }
            }
            // initial state
            qc[tid+sha] = (me >> (n-1)) & 1;
            __syncthreads();
        }

        // calculate each transition, uses GPU SIMD feature
        for (int t = 1; t < n-1; t++) {
            unsigned long long sum = 0;
            unsigned rr = row;
            for (int i = 0; i < min(k, 32); i++) {
                //sum = mod_sum(sum, qc[i+sha] * (row>>i & 1), mod);
                //sum = mod_sum64(sum, qc[i+sha] * (rr & 1), mod);
                //sum = mod_sum64(sum, qc[i+sha] * dd[i], mod);
                sum = mod_sum64(sum, qc[i+sha] & 0LL-(rr & 1), mod);
                rr >>= 1;
            }
            if (k > 32) {
                rr = row2;
                for (int i = 0; i < k-32; i++) {
                    sum = mod_sum64(sum, qc[i+32+sha] & 0ULL-(rr & 1), mod);
                    rr >>= 1;
                }
            }
            __syncthreads();
            qc[tid+sha] = sum;
            __syncthreads();
        }

        // last transition
        {
            if (!(ai[n-1] >> at & 1)) qc[tid+sha] = 0;
            __syncthreads();
            unsigned long long count = 0;
            for (int i = 0; i < k; i++) {
                count = mod_sum64(count, qc[i+sha], mod);
            }
            //if (tid==0) printf("[%d:%d],", s, count);
            if (runs + bid < work) {
                total = mod_sum64(count, total, mod);
            }
        }
        // get next work
        unsigned bit = s & (-s);
        s += bit;
        s |= mask >> __popcll(s);
        __syncthreads();
    }
    if (tid == 0) {
        // output total for this block
        ret[bid] = total;
    }
}

int n;
int adj[64*64];
unsigned part[MAX_ARRAY_SIZE];
unsigned long long ret[MAX_ARRAY_SIZE];
long long nCr[65][65];

u64 getComb(long long idx, int n, int r) {
    u64 ans = 0;
    n -= 1;
    while (r > 0) {
        if (idx < nCr[n][r]) n -= 1;
        else {
            ans |= u64(1)<<(n);
            idx -= nCr[n][r];
            n -= 1;
            r -= 1;
        }
    }
    return ans;
}

void ha4(int gridSize, int blockSize, int k, int n, int work, unsigned *part, int *adj, unsigned long long *ret, unsigned long long mod) {
    dim3 bsz(k, blockSize);
    switch (k) {
#define HA4_k(k) case k: ha2<k><<<gridSize, bsz>>>(n, work, part, adj, ret, mod); break;
    HA4_k(2)
    HA4_k(3)
    HA4_k(4)
    HA4_k(5)
    HA4_k(6)HA4_k(7)HA4_k(8)HA4_k(9)HA4_k(10)
    HA4_k(11)HA4_k(12)HA4_k(13)HA4_k(14)HA4_k(15)
    HA4_k(16)HA4_k(17)HA4_k(18)HA4_k(19)HA4_k(20)
    HA4_k(21)HA4_k(22)HA4_k(23)HA4_k(24)HA4_k(25)
    HA4_k(26)HA4_k(27)HA4_k(28)HA4_k(29)HA4_k(30)
    HA4_k(31)HA4_k(32)
#undef HA4_k
    }
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
    }
}

int main() {
    int *gpu_adj;
    unsigned *gpu_part;
    unsigned long long *gpu_ret;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) adj[i*n+j] = rand()>>5&1;
        }
    }
    for (int i = 0; i < n; i++) {
        char op;
        for (int j = 0; j < n; j++) {
            if (scanf(" %c", &op) == 1 && i != j) {
                adj[i*n+j] = op == '1';
            }
        }
    }
    for (int i = 0; i <= 64; i++) {
        nCr[i][0] = nCr[i][i] = 1;
        for (int j = 1; j < i; j++) nCr[i][j] = nCr[i-1][j-1] + nCr[i-1][j];
    }
    cudaMalloc(&gpu_part, sizeof part);
    cudaMalloc(&gpu_adj, sizeof adj);
    cudaMalloc(&gpu_ret, sizeof ret);

    cudaMemcpy(gpu_adj, adj, sizeof adj, cudaMemcpyHostToDevice);
    unsigned long long ans = 0;
    unsigned long long mod = 0;
    for (int k = 1; k <= n-1; k++) {
        int wo = nCr[n-1][k];
        int blockSize = wo;
        if (blockSize > MAX_BLOCK_SIZE / k) blockSize = MAX_BLOCK_SIZE / k;
        int gridSize = wo / blockSize;
        if (blockSize * gridSize > MAX_ARRAY_SIZE) gridSize = MAX_ARRAY_SIZE / blockSize;
        int totSize = blockSize * gridSize;
        fprintf(stderr, "block size = (%d,%d,1) grid size = (%d,1,1)\n", k, blockSize, gridSize);

        //for (int j = 0; j < wo; j++) printf("%d,", getComb(j, n-1, k));

        for (int j = 0; j < totSize; j++) {
            int step = wo / totSize * j;
            if (j < wo % totSize) step += j;
            else step += wo % totSize;
            //printf("step=%d\n", step);
            part[j] = getComb(step, n-1, k);
        }
        cudaMemcpy(gpu_part, part, sizeof(int) * totSize, cudaMemcpyHostToDevice);
        ha4(gridSize, blockSize, k, n, wo, gpu_part, gpu_adj, gpu_ret, mod);
        cudaDeviceSynchronize();
        cudaMemcpy(ret, gpu_ret, sizeof(long long) * totSize, cudaMemcpyDeviceToHost);
        unsigned long long sum = 0;
        for (int j = 0; j < totSize; j++) {
            sum = mod_sum64(sum, ret[j], 0);
        }
        //printf("sum = %u\n", sum);
        if ((n-k)%2 == 1) ans = mod_sum64(ans, sum, mod);
        else if (sum != 0) ans = mod_sum64(ans, mod-sum, mod);
    }
    printf("ans = %llu\n", ans);
    cudaFree(gpu_ret);
    cudaFree(gpu_adj);
    cudaFree(gpu_part);
    return 0;
}
