// test cuda launch speed in device
#include<stdio.h>
#include<cstring>

__global__ void emptyKernel() {
}

__global__ void cdpKernel() {
    emptyKernel<<<1,1>>>();
    emptyKernel<<<1,1>>>();
    emptyKernel<<<1,1>>>();
    emptyKernel<<<1,1>>>();
    emptyKernel<<<1,1>>>();
    emptyKernel<<<1,1>>>();
    emptyKernel<<<1,1>>>();
    emptyKernel<<<1,1>>>();
    emptyKernel<<<1,1>>>();
}

int main(int argc, char *argv[]) {
    bool sync = false;
    if (argc >= 2) {
        if (strcmp(argv[1], "sync") == 0) {
            sync = true;
        }
    }
    if (sync) {
        puts("use cudaDeviceSynchronize");
    }
    else {
        puts("no cudaDeviceSynchronize");
    }
    for (int j=0;j<10;j++){
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1, 0);
    for (int i = 0; i < 10000; i++) {
        cdpKernel<<<1,1>>>();
        if (sync) cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    cudaEventRecord(event2, 0);
    cudaEventSynchronize(event2);
    float timeMs;
    cudaEventElapsedTime(&timeMs, event1, event2);
    printf("time %fms (%f launches per sec)\n", timeMs, 1e5*1000.0/timeMs);
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    }
}
