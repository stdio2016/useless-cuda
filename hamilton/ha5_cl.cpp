// counting Hamilton cycle, OpenCL acceleration
#include<stdio.h>
#include<stdlib.h>
#include <cstring>
#include <CL/opencl.h>
#define MAX_BLOCK_SIZE 256
#define MAX_ARRAY_SIZE (1024*8)
#define WORK_LIMIT 10000000
typedef unsigned long long u64;

// these are for generating kernels
// because opencl does not support C++ template
#define STRINGIFY_(...) #__VA_ARGS__
#define STRINGIFY(...) STRINGIFY_(__VA_ARGS__)
#define ha2_(i) ha2_ ## i
#define ha2_name_gen(k) ha2_(k)
const char *kernelCode[2] = {
R"====(typedef unsigned long u64;
#define threadIdx_x get_local_id(0)
#define threadIdx_y get_local_id(1)
#define blockDim_x get_local_size(0)
#define blockDim_y get_local_size(1)
#define blockIdx_x get_group_id(0)
#define gridDim_x get_num_groups(0)
u64 mod_sum64(u64 a, u64 b, u64 mod) {
    u64 c = a+b;
    return c >= mod ? c-mod : c;
}
__kernel void ha2_1(int n, int work, __global unsigned *part, __global int *adj, __global u64 *ret, u64 mod) {
    const int bid = threadIdx_y + blockIdx_x * blockDim_y;
    ret[bid] = 0;
}
)===="
#define k 2
#include "ha5.cl"
#undef k
"\n"
#define k 3
#include "ha5.cl"
#undef k
"\n"
#define k 4
#include "ha5.cl"
#undef k
"\n"
#define k 5
#include "ha5.cl"
#undef k
"\n"
#define k 6
#include "ha5.cl"
#undef k
"\n"
#define k 7
#include "ha5.cl"
#undef k
"\n"
#define k 8
#include "ha5.cl"
#undef k
"\n"
#define k 9
#include "ha5.cl"
#undef k
"\n"
#define k 10
#include "ha5.cl"
#undef k
"\n"
#define k 11
#include "ha5.cl"
#undef k
"\n"
#define k 12
#include "ha5.cl"
#undef k
"\n"
#define k 13
#include "ha5.cl"
#undef k
"\n"
#define k 14
#include "ha5.cl"
#undef k
"\n"
#define k 15
#include "ha5.cl"
#undef k
"\n"
,
#define k 16
#include "ha5.cl"
#undef k
"\n"
#define k 17
#include "ha5.cl"
#undef k
"\n"
#define k 18
#include "ha5.cl"
#undef k
"\n"
#define k 19
#include "ha5.cl"
#undef k
"\n"
#define k 20
#include "ha5.cl"
#undef k
"\n"
#define k 21
#include "ha5.cl"
#undef k
"\n"
#define k 22
#include "ha5.cl"
#undef k
"\n"
#define k 23
#include "ha5.cl"
#undef k
"\n"
#define k 24
#include "ha5.cl"
#undef k
"\n"
#define k 25
#include "ha5.cl"
#undef k
"\n"
#define k 26
#include "ha5.cl"
#undef k
"\n"
#define k 27
#include "ha5.cl"
#undef k
"\n"
#define k 28
#include "ha5.cl"
#undef k
"\n"
#define k 29
#include "ha5.cl"
#undef k
"\n"
#define k 30
#include "ha5.cl"
#undef k
"\n"
#define k 31
#include "ha5.cl"
#undef k
"\n"
};

u64 mod_sum64(u64 a, u64 b, u64 mod) {
    u64 c = a+b;
    return c >= mod ? c-mod : c;
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

// OpenCL context
int good = 0;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernels[64] = {0};
cl_mem gpu_adj, gpu_part, gpu_ret;

int initCL() {
    // get platform
    cl_uint num = 1;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, &num);
    if (err != CL_SUCCESS || num < 1) {
        fprintf(stderr, "unable to get platform\n");
        return 0;
    }

    // get device
    num = 1;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);
    if (err != CL_SUCCESS || num < 1) {
        fprintf(stderr, "unable to get device ID\n");
        return 0;
    }

    // create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "unable to create context\n");
        return 0;
    }
    good = 1;

    // create command queue
    queue = clCreateCommandQueue(context, device, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "unable to create command queue\n");
        return 0;
    }
    good = 2;
    return 1;
}

cl_program loadKernel() {
    // compile program
    size_t lens[2] = {strlen(kernelCode[0]), strlen(kernelCode[1])};
    cl_int err;
    cl_program prog = clCreateProgramWithSource(context, 2, kernelCode, lens, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "cannot create program\n");
        if (err == CL_INVALID_CONTEXT) fprintf(stderr, "CL_INVALID_CONTEXT\n");
        if (err == CL_INVALID_VALUE) fprintf(stderr, "CL_INVALID_VALUE\n");
        if (err == CL_OUT_OF_HOST_MEMORY) fprintf(stderr, "CL_OUT_OF_HOST_MEMORY\n");
        return 0;
    }
    err = clBuildProgram(prog, 0, NULL, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "program  has errors:\n");
        size_t len;
        err = clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        if (err != CL_SUCCESS) return 0;
        char *code = new char[len];
        err = clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, len, code, &len);
        if (err == CL_SUCCESS) {
            fprintf(stderr, "%s", code);
        }
        clReleaseProgram(prog);
        return 0;
    }
    return prog;
}

void release() {
    if (good >= 7) clReleaseMemObject(gpu_ret);
    if (good >= 6) clReleaseMemObject(gpu_part);
    if (good >= 5) clReleaseMemObject(gpu_adj);
    if (good >= 4) {
        for (int i = 2; i < 64; i++) {
            if (kernels[i]) clReleaseKernel(kernels[i]);
        }
    }
    if (good >= 3) clReleaseProgram(program);
    if (good >= 2) clReleaseCommandQueue(queue);
    if (good >= 1) clReleaseContext(context);
}

int main() {
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
    
    atexit(release);
    if (!initCL()) return 1;
    program = loadKernel();
    if (!program) return 1;
  	good = 3;
    
    cl_int err1, err2;
    for (int i = 2; i < 32; i++) { // TODO support more n
        char name[20] = "ha2_xx";
        sprintf(name, "ha2_%d", i);
        kernels[i] = clCreateKernel(program, name, &err1);
        if (err1 != CL_SUCCESS) {
            fprintf(stderr, "unable to create kernel %d\n", i);
            return 1;
        }
    }
    good = 4;
    
    gpu_adj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof adj, NULL, &err2);
    if (err2 != CL_SUCCESS) { fprintf(stderr, "failed to create adj buffer\n"); return 1; }
    good = 5;

    gpu_part = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof part, NULL, &err2);
    if (err2 != CL_SUCCESS) { fprintf(stderr, "failed to create part buffer\n"); return 1; }
    good = 6;
    
    gpu_ret = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof ret, NULL, &err2);
    if (err2 != CL_SUCCESS) { fprintf(stderr, "failed to create ret buffer\n"); return 1; }
    good = 7;
    
    clEnqueueWriteBuffer(queue, gpu_adj, CL_TRUE, 0, sizeof adj, adj, 0, NULL, NULL);
    unsigned long long ans = 0;
    unsigned long long mod = 0;
    for (int k = 1; k <= n-1; k++) {
        long long works = nCr[n-1][k];
        while (works > 0) {
            long long wo = works;
            if (works > WORK_LIMIT) wo = WORK_LIMIT;
            works -= wo;
            int work = wo;
            // split too big work
            int blockSize = wo;
            if (blockSize > MAX_BLOCK_SIZE / k) blockSize = MAX_BLOCK_SIZE / k;
            int gridSize = wo / blockSize;
            if (blockSize * gridSize > MAX_ARRAY_SIZE) gridSize = MAX_ARRAY_SIZE / blockSize;
            int totSize = blockSize * gridSize;
            fprintf(stderr, "block size = (%d,%d,1) grid size = (%d,1,1) work = %lld\n", k, blockSize, gridSize, wo);

            for (int j = 0; j < totSize; j++) {
                int step = wo / totSize * j;
                if (j < wo % totSize) step += j;
                else step += wo % totSize;
                part[j] = getComb(works + step, n-1, k);
            }
            clEnqueueWriteBuffer(queue, gpu_part, CL_TRUE, 0, sizeof(int) * totSize, part, 0, NULL, NULL);
            cl_kernel kernel = kernels[k];
            if (!kernel) continue;
            clSetKernelArg(kernel, 0, sizeof(int), &n);
            clSetKernelArg(kernel, 1, sizeof(int), &work);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &gpu_part);
            clSetKernelArg(kernel, 3, sizeof(cl_mem), &gpu_adj);
            clSetKernelArg(kernel, 4, sizeof(cl_mem), &gpu_ret);
            clSetKernelArg(kernel, 5, sizeof(u64), &mod);
            size_t work_offset[2] = {0, 0};
        		size_t work_size[2] = {(size_t)(gridSize*k), (size_t)blockSize};
        		size_t local_size[2] = {(size_t)k, (size_t)blockSize};
            err1 = clEnqueueNDRangeKernel(queue, kernel, 2, work_offset, work_size, local_size, 0, NULL, NULL);
            if (err1 != CL_SUCCESS) fprintf(stderr, "failed to launch kernel %d\n", err1);
            err2 = clFinish(queue);
            if (err2 != CL_SUCCESS) fprintf(stderr, "failed to finish queue %d\n", err2);
            err1 = clEnqueueReadBuffer(queue, gpu_ret, CL_TRUE, 0, sizeof(long long) * totSize, ret, 0, NULL, NULL);
            if (err1 != CL_SUCCESS) fprintf(stderr, "failed to read buffer\n");
            unsigned long long sum = 0;
            for (int j = 0; j < totSize; j++) {
                sum = mod_sum64(sum, ret[j], 0);
            }
            if ((n-k)%2 == 1) ans = mod_sum64(ans, sum, mod);
            else if (sum != 0) ans = mod_sum64(ans, mod-sum, mod);
        }
    }
    printf("ans = %llu\n", ans);
    return 0;
}
