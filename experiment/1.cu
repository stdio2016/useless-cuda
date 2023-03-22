// get cuda max hardware concurrency
// by stdio2016 2023-03-18
#include<cuda.h>
#include<stdio.h>
__device__ int current_concurrency = 0;
__device__ void waitClockGpu(int time) {
    long long t0 = clock64();
    while (clock64() - t0 < time) {
        ;
    }
}
__global__ void concurrency_test(__global__ int *max_concurrency) {
    atomicAdd(&current_concurrency, 1);
    waitClockGpu(1000000);
    atomicMax(max_concurrency, current_concurrency);
    waitClockGpu(1000000);
    atomicAdd(&current_concurrency, -1);
}
int main(int a, char*b[]){
    int *max_concurrency;
    cudaMalloc(&max_concurrency, sizeof(int));
    int num_streams = 0;
    printf("number of streams: ");
    scanf("%d", &num_streams);
    cudaStream_t *ts = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&ts[i]);
    }
    for (int i = 0; i < num_streams; i++) {
        for (int j = 0; j < 100; j++) {
            concurrency_test<<<100, 256, 0, ts[i]>>>(max_concurrency);
        }
        //cudaStreamQuery(ts[i]);
        printf("stream %d sent\n", i);
    }
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(ts[i]);
        printf("stream %d synchronized\n", i);
    }
    int num = 0;
    cudaMemcpy(&num, max_concurrency, sizeof(int), cudaMemcpyDeviceToHost);
    printf("max concurrency: %d\n", num);
    
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(ts[i]);
    }
    delete[] ts;
    cudaFree(max_concurrency);
}
