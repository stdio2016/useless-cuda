#include <stdio.h>

int main() {
    long long kb = 0;
    printf("how many kb of pinned memory: ");
    scanf("%lld", &kb);
    void *dat;
    cudaError_t status = cudaMallocHost(&dat, (size_t)kb * 1024, cudaHostAllocPortable);
    printf("status: %d\n", status);
    cudaFreeHost(dat);
}
