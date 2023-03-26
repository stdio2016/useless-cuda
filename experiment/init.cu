#include <stdio.h>
#include<chrono>
int main() {
    auto t0 = std::chrono::steady_clock::now();
    cudaEvent_t a;
    cudaEventCreate(&a);
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = t1 - t0;
    printf("init takes %fs\n", diff.count());
}
