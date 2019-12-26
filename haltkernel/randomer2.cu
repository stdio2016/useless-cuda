#include <curand_kernel.h>
#include <stdio.h>

__global__ void randomer(int *out, volatile int *flag) {
    int cnt = 0, all = 0;
    curandState_t state;
    curand_init(123, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    for (int i = 0; i < 2147483; i++) {
        for (int j = 0; j < 1000; j++) {
            float x = curand_uniform(&state);
            float y = curand_uniform(&state);
            if (x*x + y*y <= 1.0f) cnt += 1;
        }
        all += 1000;
        if (*flag < i) break;
    }
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    out[id] = cnt;
    out[id + blockDim.x * gridDim.x] = all;
}

int *gpu_out, *out;
int *gpu_flag, *flag;

int main() {
    cudaFree(0);
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    fprintf(stderr, "Cuda device count: %d\n", deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No cuda devices!\n");
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = deviceProp.maxThreadsPerBlock;
    int gridSize = deviceProp.multiProcessorCount;
    fprintf(stderr, "block size: %d  grid size: %d\n", blockSize, gridSize);

    out = new int[blockSize * gridSize * 2];
    cudaMalloc(&gpu_out, sizeof(int) * blockSize * gridSize * 2);
    cudaMalloc(&gpu_flag, sizeof(int) * 4);
    cudaMallocHost(&flag, sizeof(int) * 4, cudaHostAllocDefault); // 一定要用 pinned memory
    flag[0] = 21474836;
    cudaMemcpy(gpu_flag, flag, sizeof(int) * 4, cudaMemcpyHostToDevice);
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    fprintf(stderr, "The program will run until you press Enter\n");
    randomer<<<gridSize, blockSize, 0, s1>>>(gpu_out, gpu_flag);
    cudaStreamQuery(s1); // 少了這行，kernel會等到cudaStreamSynchronize才執行

     // 假設這裡有一個 CPU 運算
    int computation = getchar();

    // 把 GPU 停下來
    flag[0] = computation;
    cudaMemcpyAsync(gpu_flag, flag, sizeof(int) * 4, cudaMemcpyHostToDevice, s2);
    cudaStreamSynchronize(s2);
    fprintf(stderr, "send flag\n");
    cudaStreamSynchronize(s1);
    fprintf(stderr, "finish working\n");

    // 取得目前 GPU 運行結果
    cudaMemcpy(out, gpu_out, sizeof(int) * blockSize * gridSize * 2, cudaMemcpyDeviceToHost);
    long long cnt = 0, all = 0;
    for (int i = 0; i < blockSize * gridSize; i++) {
        cnt += out[i];
    }
    for (int i = blockSize * gridSize; i < blockSize * gridSize * 2; i++) {
        all += out[i];
    }
    fprintf(stderr, "all = %lld cnt = %lld pi = %.10f\n", all, cnt, (double)cnt / all * 4.0);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(gpu_flag);
    cudaFreeHost(flag);
    cudaFree(gpu_out);
    delete[] out;
}