#include<stdio.h>
typedef signed i32;
typedef unsigned u32;
typedef unsigned long long u64;
__global__ void findlight(int n, int m, u64 *work, u32 pitch, u32 *result, volatile int *flag) {
    __shared__ u32 hasWork;
    int tid = threadIdx.x;
    hasWork = blockDim.x;
    __syncthreads();
    int idx = tid;
    u64 lo = work[idx];
    u64 hi = work[idx + pitch];
    u32 mask1 = n > 32 ? (2U<<n-33)-1 : 0U;
    u32 mask2 = n > 32 ? ~0U : (2U<<n-1)-1;
    for (;;) {
        int ans = 0;
        for (u64 row = lo; row < hi; row++) {
            u32 up1 = 0, up2 = 0;
            u32 tmp1 = row>>31, tmp2 = row<<1;
            for (int i = 0; i < m; i++) {
                u32 left1 = tmp1<<1 | tmp2>>31;
                u32 left2 = tmp2<<1;
                u32 right1 = tmp1>>1;
                u32 right2 = tmp1<<31 | tmp2>>1;
                u32 down1 = ~(tmp1 ^ left1 ^ right1 ^ up1) & mask1;
                u32 down2 = ~(tmp2 ^ left2 ^ right2 ^ up2) & mask2;
                up1 = tmp1; up2 = tmp2;
                tmp1 = down1; tmp2 = down2;
            }
            if (tmp1 == 0 && tmp2 == 0) ans += 1;
            up1 = up2 = 0;
            tmp1 = row>>31, tmp2 = row<<1|1;
            for (int i = 0; i < m; i++) {
                u32 left1 = tmp1<<1 | tmp2>>31;
                u32 left2 = tmp2<<1;
                u32 right1 = tmp1>>1;
                u32 right2 = tmp1<<31 | tmp2>>1;
                u32 down1 = ~(tmp1 ^ left1 ^ right1 ^ up1) & mask1;
                u32 down2 = ~(tmp2 ^ left2 ^ right2 ^ up2) & mask2;
                up1 = tmp1; up2 = tmp2;
                tmp1 = down1; tmp2 = down2;
            }
            if (tmp1 == 0 && tmp2 == 0) ans += 1;
        }
        result[idx] = ans;
        idx = atomicAdd(&hasWork, 1);
        if (idx >= pitch || *flag == 0) break;
        lo = work[idx];
        hi = work[idx + pitch];
    }
}
int main() {
    int n, m;
    i32 *flag;
    cudaHostAlloc(&flag, 1024 * sizeof(i32), cudaHostAllocDefault);
    printf("light out solution count\n");
    printf("enter n(1~64), m(1+): ");
    scanf("%d %d", &n, &m);
    if (n < 1 || n > 64) return 1;
    if (m < 1) return 1;
    u64 work = 100 * 1024;
    if (n <= 10) work = 1024;
    else if (n <= 16) work = (1<<n-10) * 1024;
    u64 all = u64(1)<<(n-1);
    u64 *work_arr = new u64[work*2];
    u32 *result_arr = new u32[work];
    u64 step = all / work, rem = all % work;
    for (int i = 0; i < work; i++) {
        u64 from = step * i + min(u64(i), rem);
        u64 to = step * (i+1) + min(u64(i+1), rem);
        work_arr[i] = from;
        work_arr[i+work] = to;
    }
    u64 *gpu_work_arr;
    u32 *gpu_result;
    i32 *gpu_flag;
    cudaMalloc(&gpu_work_arr, work * 2 * sizeof(u64));
    cudaMalloc(&gpu_result, work * sizeof(u32));
    cudaMalloc(&gpu_flag, 1024 * sizeof(i32));
    for (int i = 0; i < 1024; i++) flag[i] = 1;
    cudaMemcpy(gpu_work_arr, work_arr, work * 2 * sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_flag, flag, 1024 * sizeof(i32), cudaMemcpyHostToDevice);
    findlight<<<1, 1024>>>(n, m, gpu_work_arr, work, gpu_result, gpu_flag);
    cudaMemcpy(result_arr, gpu_result, work * sizeof(u32), cudaMemcpyDeviceToHost);
    int sum = 0;
    for (int i = 0; i < work; i++) {
        sum += result_arr[i];
    }
    printf("solution count = %d\n", sum);
    cudaFree(gpu_work_arr);
    cudaFree(gpu_result);
    cudaFree(gpu_flag);
    cudaFreeHost(flag);
}