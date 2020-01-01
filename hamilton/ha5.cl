// Warning! this kernel is intended to be included in C++ code, not loaded directly
// define k before including this
// OpenCL doesn't have "long long". It is actually "long"
R"====(
#define k )====" STRINGIFY(k) "\n"
STRINGIFY(__kernel void ha2_name_gen(k)(int n, int work, __global unsigned *part, __global int *adj, __global u64 *ret, u64 mod) {)
R"====(
    __local unsigned long qc[1024]; // transition count
    __local unsigned long ai[64]; // adjacency matrix as bitset
    //const int k = blockDim.x;
    const int tid = threadIdx_x;
    const int bid = threadIdx_y + blockIdx_x * blockDim_y;
    const int sha = threadIdx_y * k;
    const int gridSize = blockDim_y * gridDim_x;
    unsigned long s = part[bid];
    unsigned long mask = (1UL<<k) - 1;
    unsigned long total = 0;

    // fetch adjacency matrix
    for (int i = tid+sha; i < n; i += blockDim_y * k) {
        unsigned long aa = 0;
        for (int j = 0; j < n; j++) {
            aa = aa | (u64)(adj[i * n + j]) << j;
        }
        ai[i] = aa;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int runs = 0; runs < work; runs += gridSize) {
        unsigned at;
        {
            unsigned long row = s;
            for (int i = 0; i < tid; i++) {
                row = row & (row-1);
            }
            at = 63 - clz(row & -row); // __ffsll(row) - 1
        }
        // making row "long long" would make program 3x slow, so I use 2 unsigned int
        unsigned row = 0, row2 = 0;
        {
            // build transition table
            unsigned long me = ai[at];
            for (int i = n-2; i >= 0; i--) {
                if (s>>i & 1) {
                    row2 = row2 << 1 | row >> 31;
                    row = row + row + (me>>i & 1);
                }
            }
            // initial state
            qc[tid+sha] = (me >> (n-1)) & 1;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // calculate each transition, uses GPU SIMD feature
        for (int t = 1; t < n-1; t++) {
            unsigned long sum = 0;
            unsigned rr = row;
            for (int i = 0; i < min(k, 32); i++) {
                //sum = mod_sum(sum, qc[i+sha] * (row>>i & 1), mod);
                //sum = mod_sum64(sum, qc[i+sha] * (rr & 1), mod);
                //sum = mod_sum64(sum, qc[i+sha] * dd[i], mod);
                sum = mod_sum64(sum, qc[i+sha] & 0L-(rr & 1), mod);
                rr >>= 1;
            }
            if (k > 32) {
                rr = row2;
                for (int i = 0; i < k-32; i++) {
                    sum = mod_sum64(sum, qc[i+32+sha] & 0UL-(rr & 1), mod);
                    rr >>= 1;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            qc[tid+sha] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // last transition
        {
            if (!(ai[n-1] >> at & 1)) qc[tid+sha] = 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            unsigned long count = 0;
            for (int i = 0; i < k; i++) {
                count = mod_sum64(count, qc[i+sha], mod);
            }
            //if (tid==0) printf("[%d:%d],", s, count);
            if (runs + bid < work) {
                total = mod_sum64(count, total, mod);
            }
        }
        // get next work
        u64 bit = s & (-s);
        s += bit;
        s |= mask >> popcount(s);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        // output total for this block
        ret[bid] = total;
    }
}
#undef k
)===="
