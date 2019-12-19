#include<stdio.h>

// any 2 <= mod <= 2^31 should work
__host__ __device__ unsigned mod_sum(unsigned a, unsigned b, unsigned mod) {
    unsigned c = a+b;
    return c >= mod ? c-mod : c;
}

// each block solves a case
// block size must be n - 1
__global__ void my_hamilton(int n, int *adj, int *ret, unsigned int mod) {
    __shared__ unsigned qc[512];
    __shared__ unsigned a_n_1[32]; // adj[n-1][i]
    int tid = threadIdx.x; // logical thread index
    int bid = blockIdx.x * blockDim.y + threadIdx.y; // logical block index
    int sha = threadIdx.y * (n-1); // because logical thread blocks are sharing real block
    int gridSize = blockDim.y * gridDim.x;// logical grid size

    // prefetch
    unsigned a_i = 0;
    for (int i = 0; i < n; i++) {
        a_i = a_i | adj[tid*n + i]<<i;
    }
    if (threadIdx.y == 0) a_n_1[tid] = adj[(n-1)*n + tid];

    unsigned total = 0;
    unsigned times = ((1U<<(n-1))-1) / gridSize + 1;
    if ((times-1) * gridSize + blockIdx.y * blockDim.x >= (1U<<(n-1))) times -= 1;
    // test each case in this block
    for (unsigned _t = 0; _t < times; _t++) {
        unsigned s = _t * gridSize + bid;
        // active means this thread is selected
        unsigned active = s>>tid & 1;

        // first transition
        qc[tid + sha] = active * (a_i>>(n-1) & 1);
        unsigned row = active * a_i;
        __syncthreads();

        // calculate each transition, uses GPU SIMD feature
        for (int t = 1; t < n-1; t++) {
            unsigned sum = 0;
            for (int i = 0; i < n-1; i++) {
                sum = mod_sum(sum, qc[i + sha] * (row>>i & 1), mod);
            }
            __syncthreads();
            qc[tid + sha] = sum;
            __syncthreads();
        }

        // last transition
        unsigned count = 0;
        for (int i = 0; i < n-1; i++) {
            count = mod_sum(count, qc[i + sha] * a_n_1[i], mod);
        }
        count *= s < (1U<<(n-1));

        // adjust sign for inclusion-exclusion principle
        int sign = (n - __popc(s)) & 1;
        unsigned count_with_sign = sign ? count : (count ? mod-count : 0);
        total = mod_sum(total, count_with_sign, mod);
        //if(tid==0) printf("%ds=%d total=%d\n",active,s, count);
        __syncthreads();
    }
    if (tid == 0) {
        // output total for this block
        ret[bid] = total;
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

#define showCudaError(errorcode) showError(errorcode, __FILE__, __LINE__)
void showError(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "<error> at %s:%d: %s\n", file, line, cudaGetErrorString(code));
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

    // decide block size and grid size
    unsigned works = 1U<<(n-1);
    int grid_size = 1, block_size_y = 1;
    int sum_size = 128;
    block_size_y = 512 / (n-1);
    if (block_size_y > works) block_size_y = works;
    grid_size = (works-1) / block_size_y + 1;
    if (grid_size > 65536) grid_size = 65536;
    int works_per_thread = (works-1) / (grid_size * block_size_y) + 1;
    grid_size = (works-1) / (works_per_thread * block_size_y) + 1;

    int logical_grid_size = grid_size * block_size_y;
    works_per_thread = (works-1) / logical_grid_size + 1;

    // allocate memory
    int *gpu_a, *gpu_ans, *gpu_sum;
    showCudaError(cudaMalloc(&gpu_a, sizeof a));
    showCudaError(cudaMalloc(&gpu_ans, sizeof(int) * logical_grid_size)); // only resides in GPU!
    showCudaError(cudaMalloc(&gpu_sum, sizeof(int) * sum_size));
    showCudaError(cudaMemcpy(gpu_a, a, sizeof a, cudaMemcpyHostToDevice));

    // show work info
    printf("grid size: %d\n", grid_size);
    printf("block size: %d,%d\n", n-1, block_size_y);
    printf("logical grid size: %d\n", logical_grid_size);
    printf("works per thread: %d\n", works_per_thread);

    // run!
    for (int i = 1; i < argc; i++) {
        unsigned mod = 0;
        sscanf(argv[i], "%u", &mod);
        
        my_hamilton<<<grid_size, dim3(n-1, block_size_y)>>>(n, gpu_a, gpu_ans, mod);
        showCudaError(cudaDeviceSynchronize());
        sum_all<<<sum_size, 256>>>(logical_grid_size, gpu_ans, gpu_sum, mod);
        showCudaError(cudaDeviceSynchronize());
        showCudaError(cudaMemcpy(sum, gpu_sum, sizeof(int) * sum_size, cudaMemcpyDeviceToHost));
        unsigned ans = 0;
        for (int j = 0; j < sum_size; j++) ans = mod_sum(ans, sum[j], mod);
        printf("%u\n", ans);
    }
}
