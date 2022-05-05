#include <metal_atomic>

struct SubP {
    uint mid, diag1, diag2;
    uint ss;
};

kernel void nqueen_kern(device int *lv_ptr,
                        device const uint canplace[],
                        device const struct SubP *works,
                        device int *workCount_ptr,
                        device metal::atomic_uint *flag,
                        device uint* result,
                        threadgroup uint s0[12][256],
                        uint tid [[thread_position_in_threadgroup]],
                        uint gid [[thread_position_in_grid]])
{
    int lv = *lv_ptr;
    uint workCount = *workCount_ptr;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (gid >= workCount) {
        result[gid] = 0;
        return;
    }

    int i = lv-1;
    uint sol = 0;
    uint my = gid;
    uint mid = works[my].mid;
    uint diag1 = works[my].diag1, diag1r = 0;
    uint diag2 = works[my].diag2, diag2r = 0;
    s0[i][tid] = canplace[i] & ~(mid | diag1 | diag2);
    uint runtime = 0;
    for (; my < workCount; ) {
        uint can = s0[i][tid];
        uint s = can & -can;
        if (can) {
            mid += s;
            diag1 += s;
            diag1r = diag1r << 1 | diag1 >> 31;
            diag1 <<= 1;
            diag2 += s;
            diag2r = diag2r >> 1 | diag2 << 31;
            diag2 >>= 1;
            i -= 1;
            can = canplace[i] & ~(mid | diag1 | diag2);
            s0[i][tid] = can;
        }
        if (can && i == 0) sol += 1, can = 0;
        if (!can && i+1<lv) {
            i += 1;
            can = s0[i][tid];
            s = can & -can;
            mid -= s;
            diag1 = diag1r << 31 | diag1 >> 1;
            diag1r >>= 1;
            diag1 -= s;
            diag2 = diag2r >> 31 | diag2 << 1;
            diag2r <<= 1;
            diag2 -= s;
            s0[i][tid] = can - s;
        }
        if (!can && i == lv-1) {
            my = atomic_fetch_add_explicit(flag, 1, metal::memory_order_relaxed);
            if (my < workCount) {
                mid = works[my].mid;
                diag1 = works[my].diag1, diag1r = 0;
                diag2 = works[my].diag2, diag2r = 0;
                s0[i][tid] = canplace[i] & ~(mid | diag1 | diag2);
            }
        }
        runtime++;
    }
    result[gid] = sol;
}
