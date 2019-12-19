#include<stdio.h>

// any 2 <= mod <= 2^31 should work
__host__ __device__ unsigned mod_sum(unsigned a, unsigned b, unsigned mod) {
    unsigned c = a+b;
    return c >= mod ? c-mod : c;
}

// each block solves a case
// block size must be power of 2 and <= 2^(n-1)
// thread size must be n - 1
__global__ void my_hamilton(int n, int *adj, int *ret, unsigned int mod) {
    __shared__ unsigned qc[31];
    __shared__ unsigned a_n_1[31]; // adj[n-1][i]
    int tid = threadIdx.x;
    int lv = 31 - __clz(gridDim.x);
    unsigned s = blockIdx.x; // case as a bit field
    unsigned total = 0;

    // prefetch
    unsigned a_i = 0;
    for (int i = 0; i < n; i++) {
        a_i = a_i | adj[tid*n + i]<<i;
    }
    a_n_1[tid] = adj[(n-1)*n + tid];

    // test each case in this block
    for (unsigned stride = 0; stride < 1U<<(n-1-lv); stride++, s += 1U<<lv) {
        // active means this thread is selected
        unsigned active = s>>tid & 1;

        // first transition
        qc[tid] = active * (a_i>>(n-1) & 1);
        unsigned row = active * a_i;
        __syncthreads();

        // calculate each transition, uses GPU SIMD feature
        for (int t = 1; t < n-1; t++) {
            unsigned sum = 0;
            for (int i = 0; i < n-1; i++) {
                sum = mod_sum(sum, qc[i] * (row>>i & 1), mod);
            }
            __syncthreads();
            qc[tid] = sum;
        }

        // last transition
        unsigned count = 0;
        for (int i = 0; i < n-1; i++) {
            count = mod_sum(count, qc[i] * a_n_1[i], mod);
        }

        // adjust sign for inclusion-exclusion principle
        int sign = (n - __popc(s)) & 1;
        unsigned count_with_sign = sign ? count : (count ? mod-count : 0);
        total = mod_sum(total, count_with_sign, mod);
    }
    if (tid == 0) {
        // output total for this block
        ret[blockIdx.x] = total;
    }
}

// thread size must be >= 64 and power of 2
__global__ void sum_all(int n, int *data, int *sum, unsigned mod) {
    __shared__ int tmp_sum[1024];
    int blockSize = blockDim.x;
    int stride = gridDim.x * blockSize;
    int id = threadIdx.x;
    int i = id + blockSize * blockIdx.x;

    // sum part of data
    tmp_sum[id] = 0;
    while (i < n) {
        tmp_sum[id] = mod_sum(tmp_sum[id], data[i], mod);
        i += stride;
    }
    __syncthreads();

    // merge threads
    if (blockSize >= 1024) {
        if (id < 512) tmp_sum[id] = mod_sum(tmp_sum[id], tmp_sum[id + 512], mod);
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (id < 256) tmp_sum[id] = mod_sum(tmp_sum[id], tmp_sum[id + 256], mod);
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (id < 128) tmp_sum[id] = mod_sum(tmp_sum[id], tmp_sum[id + 128], mod);
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (id < 64) tmp_sum[id] = mod_sum(tmp_sum[id], tmp_sum[id + 64], mod);
        __syncthreads();
    }
    if (id < 32) {
        // now, only 1 warp is active
        volatile int *tmp = tmp_sum;
        tmp[id] = mod_sum(tmp[id], tmp[id + 32], mod);
        tmp[id] = mod_sum(tmp[id], tmp[id + 16], mod);
        tmp[id] = mod_sum(tmp[id], tmp[id + 8], mod);
        tmp[id] = mod_sum(tmp[id], tmp[id + 4], mod);
        tmp[id] = mod_sum(tmp[id], tmp[id + 2], mod);
        tmp[id] = mod_sum(tmp[id], tmp[id + 1], mod);
    }
    // write back to global memory
    if (id == 0) {
        sum[blockIdx.x] = tmp_sum[0];
    }
}

int n, a[32*32], sum[1<<7];
int main(int argc, char *argv[]) {
    if (scanf("%d", &n) != 1) return 1;
    if (n < 3 || n > 32) return 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int aij = 1; scanf("%d", &aij);
            if (i == j) a[i*n+j] = 0;
            else a[i*n+j] = aij;
        }
    }

    int block_size = 1;
    const int MAX_BLOCK_LV = 16;
    if (n <= MAX_BLOCK_LV) block_size = 1<<(n-1);
    else block_size = 1<<MAX_BLOCK_LV;
    int sum_size = 128;

    int *gpu_a, *gpu_ans, *gpu_sum;
    cudaMalloc(&gpu_a, sizeof a);
    cudaMalloc(&gpu_ans, sizeof(int) * block_size); // only resides in GPU!
    cudaMalloc(&gpu_sum, sizeof(int) * sum_size);
    cudaMemcpy(gpu_a, a, sizeof a, cudaMemcpyHostToDevice);

    for (int i = 1; i < argc; i++) {
        unsigned mod = 0;
        sscanf(argv[i], "%u", &mod);
        my_hamilton<<<block_size, n-1>>>(n, gpu_a, gpu_ans, mod);
        sum_all<<<sum_size, 256>>>(block_size, gpu_ans, gpu_sum, mod);
        cudaDeviceSynchronize();
        cudaMemcpy(sum, gpu_sum, sizeof(int) * sum_size, cudaMemcpyDeviceToHost);
        unsigned ans = 0;
        for (int j = 0; j < sum_size; j++) ans = mod_sum(ans, sum[j], mod);
        printf("%u\n", ans);
    }
}