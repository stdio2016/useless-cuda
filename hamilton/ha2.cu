// counting Hamilton cycle, CUDA acceleration
#include<stdio.h>
#include<stdlib.h>
#define MAX_BLOCK_SIZE 1024
#define MAX_ARRAY_SIZE (1024*8)

// any 2 <= mod <= 2^31 should work
__host__ __device__ unsigned mod_sum(unsigned a, unsigned b, unsigned mod) {
    unsigned c = a+b;
    return c >= mod ? c-mod : c;
}

__global__ void ha2(int n, int work, unsigned *part, int *adj, unsigned *ret, unsigned int mod) {
    __shared__ unsigned qc[1024];
    __shared__ unsigned ai[32];
    int k = blockDim.x;
    int tid = threadIdx.x;
    int sha = threadIdx.y * k;
    int bid = threadIdx.y + blockIdx.x * blockDim.y;
    int gridSize = blockDim.y * gridDim.x;
    unsigned s = part[bid];
    unsigned mask = (1u<<k) - 1;
    unsigned total = 0;

    for (int i = tid+sha; i < n; i += blockDim.y * k) {
        unsigned aa = 0;
        for (int j = 0; j < n; j++) {
            aa = aa | adj[i * n + j] << j;
        }
        ai[i] = aa;
    }
    __syncthreads();

    for (int runs = 0; runs < work; runs += gridSize) {
        // first transition
        unsigned row = s;
        for (int i = 0; i < tid; i++) {
            row = row & (row-1);
        }
        unsigned at = __ffs(row)-1;
        row = 0;
        {
            unsigned me = ai[at];
            for (int i = n-2; i >= 0; i--) {
                if (s>>i & 1) {
                    row = row + row + (me>>i & 1);
                }
            }
            qc[tid+sha] = (me >> (n-1)) & 1;
            __syncthreads();
        }

        // calculate each transition, uses GPU SIMD feature
        for (int t = 1; t < n-1; t++) {
            unsigned sum = 0;
            for (int i = 0; i < k; i++) {
                sum = mod_sum(sum, qc[i+sha] * (row>>i & 1), mod);
            }
            __syncthreads();
            qc[tid+sha] = sum;
            __syncthreads();
        }

        // last transition
        {
            if (!(ai[n-1] >> at & 1)) qc[tid+sha] = 0;
            __syncthreads();
            unsigned count = 0;
            for (int i = 0; i < k; i++) {
                count = mod_sum(count, qc[i+sha], mod);
            }
            //if (tid==0) printf("[%d:%d],", s, count);
            if (runs + bid < work) {
                total = mod_sum(count, total, mod);
            }
        }
        unsigned bit = s & (-s);
        s += bit;
        s |= mask >> __popc(s);
        __syncthreads();
    }
    if (tid == 0) {
        // output total for this block
        ret[bid] = total;
    }
}

int n;
int adj[1024];
unsigned part[MAX_ARRAY_SIZE];
unsigned ret[MAX_ARRAY_SIZE];
int nCr[33][33];

unsigned getComb(int idx, int n, int r) {
    unsigned ans = 0;
    n -= 1;
    while (r > 0) {
        if (idx < nCr[n][r]) n -= 1;
        else {
            ans |= 1u<<(n);
            idx -= nCr[n][r];
            n -= 1;
            r -= 1;
        }
    }
    return ans;
}

int main() {
    int *gpu_adj;
    unsigned *gpu_part, *gpu_ret;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) adj[i*n+j] = rand()>>5&1;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) scanf("%d", &adj[i*n+j]);
    }
    for (int i = 0; i <= 32; i++) {
        nCr[i][0] = nCr[i][i] = 1;
        for (int j = 1; j < i; j++) nCr[i][j] = nCr[i-1][j-1] + nCr[i-1][j];
    }
    cudaMalloc(&gpu_part, sizeof part);
    cudaMalloc(&gpu_adj, sizeof adj);
    cudaMalloc(&gpu_ret, sizeof ret);

    cudaMemcpy(gpu_adj, adj, sizeof adj, cudaMemcpyHostToDevice);
    unsigned ans = 0;
    unsigned mod = 0;
    for (int k = 1; k <= n-1; k++) {
        int wo = nCr[n-1][k];
        int blockSize = wo;
        if (blockSize > MAX_BLOCK_SIZE / k) blockSize = MAX_BLOCK_SIZE / k;
        int gridSize = wo / blockSize;
        if (blockSize * gridSize > MAX_ARRAY_SIZE) gridSize = MAX_ARRAY_SIZE / blockSize;
        int totSize = blockSize * gridSize;
        printf("block size = (%d,%d,1) grid size = (%d,1,1)\n", k, blockSize, gridSize);

        //for (int j = 0; j < wo; j++) printf("%d,", getComb(j, n-1, k));

        for (int j = 0; j < totSize; j++) {
            int step = wo / totSize * j;
            if (j < wo % totSize) step += j;
            else step += wo % totSize;
            //printf("step=%d\n", step);
            part[j] = getComb(step, n-1, k);
        }
        cudaMemcpy(gpu_part, part, sizeof(int) * totSize, cudaMemcpyHostToDevice);
        ha2<<<gridSize, dim3(k, blockSize)>>>(n, wo, gpu_part, gpu_adj, gpu_ret, mod);
        cudaDeviceSynchronize();
        cudaMemcpy(ret, gpu_ret, sizeof(int) * totSize, cudaMemcpyDeviceToHost);
        unsigned sum = 0;
        for (int j = 0; j < totSize; j++) {
            sum = mod_sum(sum, ret[j], 0);
        }
        printf("sum = %u\n", sum);
        if ((n-k)%2 == 1) ans = mod_sum(ans, sum, mod);
        else if (sum != 0) ans = mod_sum(ans, mod-sum, mod);
    }
    printf("ans = %u\n", ans);
    cudaFree(gpu_ret);
    cudaFree(gpu_adj);
    cudaFree(gpu_part);
}
