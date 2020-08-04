#include <stdio.h>
#include <cmath>

//const double pi2 = 3.14159265358979323846264338327950288 * 2.0;
#ifdef __CUDA_ARCH__
__device__
#endif
const double pi2 = 3.14159265358979323846264338327950288 * 2.0;

// reference:
// Andrew Thall. Extended-Precision Floating-Point Numbers for GPU Computation.
__device__ float2 Re(float4 a) {
    return make_float2(a.x, a.y);
}

__device__ float2 Im(float4 a) {
    return make_float2(a.z, a.w);
}

__device__ float4 Cplex(float2 re, float2 im) {
    return make_float4(re.x, re.y, im.x, im.y);
}

__device__ float2 Neg(float2 a) {
    return make_float2(-a.x, -a.y);
}

// |a| > |b|
__device__ float2 Fast2Sum(float a, float b) {
    float sum = a + b;
    float r = sum - a;
    float err = b - r;
    return make_float2(sum, err);
}

__device__ float2 Sum(float a, float b) {
    float sum = a + b;
    float aa = sum - b;
    float bb = sum - aa;
    float ea = a - aa;
    float eb = b - bb;
    float err = ea + eb;
    return make_float2(sum, err);
}

__device__ float2 Sum2(float2 a, float2 b) {
    float2 sum = Sum(a.x, b.x);
    float2 err = Sum(a.y, b.y);
    sum = Fast2Sum(sum.x, sum.y + err.x);
    sum = Fast2Sum(sum.x, sum.y + err.y);
    return sum;
}

__device__ float4 CSum(float4 a, float4 b) {
    float2 x = Sum2(Re(a), Re(b));
    float2 y = Sum2(Im(a), Im(b));
    return Cplex(x, y);
}

__device__ float2 Fast2Mul(float a, float b) {
    float mul = a * b;
    return make_float2(mul, __fmaf_rn(a, b, -mul));
}

__device__ float2 Mul2(float2 a, float2 b) {
    float2 mul = Fast2Mul(a.x, b.x);
    mul.y = __fmaf_rn(a.y, b.y, mul.y);
    mul.y = __fmaf_rn(a.y, b.x, mul.y);
    mul.y = __fmaf_rn(a.x, b.y, mul.y);
    return Fast2Sum(mul.x, mul.y);
}

__device__ float4 CMul(float4 a, float4 b) {
    float2 ar = Re(a), ai = Im(a), br = Re(b), bi = Im(b);
    float2 re = Sum2(Mul2(ar, br), Neg(Mul2(ai, bi)));
    float2 im = Sum2(Mul2(ar, bi), Mul2(ai, br));
    return Cplex(re, im);
}

__global__ void fft_permute(float4 *din, float4 *dout, int width) {
   __shared__ float buf[4][32][33];
   int tx = threadIdx.x, ty = threadIdx.y;
   int bx = blockIdx.x;
   unsigned me = (ty << (width-5)) + tx;
   unsigned pos = bx << 5;
   unsigned revpos = __brev(me + pos) >> (32-width);
   float4 val = din[me + pos];
   buf[0][tx][ty] = val.x;
   buf[1][tx][ty] = val.y;
   buf[2][tx][ty] = val.z;
   buf[3][tx][ty] = val.w;
   __syncthreads();
   tx = (revpos>>(width-5)) & 31;
   ty = revpos & 31;
   val.x = buf[0][ty][tx];
   val.y = buf[1][ty][tx];
   val.z = buf[2][ty][tx];
   val.w = buf[3][ty][tx];
   unsigned mask = (1<<(width-5)) - 32;
   dout[me + (revpos & mask)] = val;
}

__global__ void fft_permute_small(float4 *din, float4 *dout, int width) {
    __shared__ float4 buf[1024+32];
    if (width < 1 || width > 10) return ;
    int tx = threadIdx.x, off;
    int bx = blockIdx.x;
    unsigned me = bx * blockDim.x + tx;
    unsigned rev = __brev(tx) >> (32-width) | tx & (~0<<width);
    float4 val = din[me];
    off = tx>>5;
    buf[tx+off] = val;
    __syncthreads();
    off = rev>>5;
    val = buf[rev+off];
    dout[me] = val;
}

__global__ void fft_transpose(float4 *din, float4 *dout, int width, int height, float4 *twiddle, unsigned mask) {
    __shared__ float4 buf[1024+32];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int x = 32 * bx;
    int y = 32 * by;
    if (tx+x < width && ty+y < height) {
        int idx = (tx + x) + (ty + y) * width;
        buf[tx + ty*33] = CMul(din[idx], twiddle[idx & mask]);
    }
    __syncthreads();
    if (ty+x < width && tx+y < height) {
        dout[(tx + y) + (ty + x) * height] = buf[ty + tx*33];
    }
}

__device__ float4 shflxor(float4 a, int mask) {
    float4 b;
    b.x = __shfl_xor_sync(0xffffffff, a.x, mask, 32);
    b.y = __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    b.z = __shfl_xor_sync(0xffffffff, a.z, mask, 32);
    b.w = __shfl_xor_sync(0xffffffff, a.w, mask, 32);
    return b;
}

__global__ void fft_a(float4 *din, float4 *dout, float4 *trig, int width) {
    int tid = threadIdx.x, bid = blockIdx.x;
    int bs = blockDim.x;
    int parity;
    float4 me, him;
    if (width < 1 || width > 10) return ;

    parity = 1 - (tid&1)*2;
    me = din[tid + bs * bid];
    him = shflxor(me, 1);
    me.x *= parity, me.y *= parity, me.z *= parity, me.w *= parity;
    me = CSum(me, him);

    for (int iter = 1; iter < min(5,width); iter++) {
        int ord = tid & ((2<<iter)-1);
        parity = 1 - (ord>>iter)*2;
        me = CMul(me, trig[ord+(2<<iter)]);
        him = shflxor(me, 1<<iter);
        me.x *= parity, me.y *= parity, me.z *= parity, me.w *= parity;
        me = CSum(me, him);
    }
    __shared__ float4 share[1024];
    for (int iter = 5; iter < width; iter++) {
        __syncthreads();
        int ord = tid & ((2<<iter)-1);
        parity = 1 - (ord>>iter)*2;
        me = CMul(me, trig[ord+(2<<iter)]);
        share[tid] = me;
        me.x *= parity, me.y *= parity, me.z *= parity, me.w *= parity;
        __syncthreads();
        him = share[tid ^ (1<<iter)];
        me = CSum(me, him);
    }
    dout[tid + bs * bid] = me;
}

__global__ void sintable(float4 *dout) {
    int Nx = blockDim.x * gridDim.x;
    int Ny = blockDim.y * gridDim.y;
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    double theta = pi2 * (1.0f/(Nx*Ny)) * (tx*ty);
    double co_ = cos(theta), si_ = sin(theta);
    float4 o;
    o.x = co_;
    o.y = co_ - o.x;
    o.z = si_;
    o.w = si_ - o.z;
    dout[tx + ty * Nx] = o;
}

int main() {
    float4 *mydata;
    float4 *din, *dout, *trig, *twiddle;
    int width = 20;
    cudaMallocHost(&mydata, (sizeof(float4)*2)<<width);
    cudaMalloc(&din, sizeof(float4)<<width);
    cudaMalloc(&dout, sizeof(float4)<<width);
    cudaMalloc(&trig, sizeof(float4) * 2048);
    cudaMalloc(&twiddle, sizeof(float4)<<width);
    for (int i = 0; i < 10; i++) {
        for (int ord = 0; ord < 1<<i; ord++) {
            mydata[ord + (2<<i)] = make_float4(1, 0, 0, 0);
        }
        for (int ord = 1<<i; ord < 2<<i; ord++) {
            double frag = (double)ord / (2<<i) - 0.5;
            double si_ = sin(pi2 * frag);
            double co_ = cos(pi2 * frag);
            float4 aa;
            aa.x = co_;
            aa.y = co_ - aa.x;
            aa.z = si_;
            aa.w = si_ - aa.z;
            mydata[ord + (2<<i)] = aa;
        }
    }
    cudaMemcpy(trig, mydata, sizeof(float4) * 2048, cudaMemcpyHostToDevice);
    sintable<<<dim3(4, 1<<(width-10)), 256>>>(twiddle);
    for (int i = 0; i < 2<<width; i++) mydata[i] = make_float4(0, 0, 0, 0);
    mydata[1].x = 1;
    cudaMemcpy(din, mydata, sizeof(float4)<<width, cudaMemcpyHostToDevice);
    if (width >= 15)
       fft_transpose<<<dim3(1<<(width-15),32), dim3(32,32)>>>(din, dout, 1<<(width-10), 1024, twiddle, 0);
    else
       fft_transpose<<<dim3(1,32), dim3(32,32)>>>(din, dout, 1<<(width-10), 1024, twiddle, 0);
    fft_permute_small<<<1<<(width-10), 1024>>>(dout, din, 10);
    fft_a<<<1<<(width-10), 1024>>>(din, dout, trig, 10);

    if (width >= 15)
        fft_transpose<<<dim3(32,1<<(width-15)), dim3(32,32)>>>(dout, din, 1024, 1<<(width-10), twiddle, (1<<width)-1);
    else
        fft_transpose<<<dim3(32,1), dim3(32,32)>>>(dout, din, 1024, 1<<(width-10), twiddle, (1<<width)-1);
    fft_permute_small<<<1<<(width-10), 1024>>>(din, dout, min(10, width-10));
    fft_a<<<1<<(width-10), 1024>>>(dout, din, trig, min(10, width-10));

    if (width >= 15)
       fft_transpose<<<dim3(1<<(width-15),32), dim3(32,32)>>>(din, dout, 1<<(width-10), 1024, twiddle, 0);
    else
       fft_transpose<<<dim3(1,32), dim3(32,32)>>>(din, dout, 1<<(width-10), 1024, twiddle, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(mydata, dout, sizeof(float4)<<width, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32; i++) {
        int y = i+32;
        printf("%f + %f i\n", mydata[y].x + mydata[y].y, mydata[y].z + mydata[y].w);
    }
    return 0;
}
