#include <metal_atomic>

struct SubP {
    uint mid, diag1, diag2;
    uint ss;
};

struct ProgressStore {
    uint i;
    uint mid;
    uint diag1, diag1r;
    uint diag2, diag2r;
    uint s0[12];
};

kernel void nqueen_kern(device int *lv_ptr,
                        device const uint canplace[],
                        device const struct SubP *works,
                        device int *workCount_ptr,
                        device metal::atomic_uint *flag,
                        device uint* result,
                        device struct ProgressStore *progress,
                        threadgroup uint s0[12][256],
                        uint tid [[thread_position_in_threadgroup]],
                        uint gid [[thread_position_in_grid]])
{
    int lv = *lv_ptr;
    int workCount = *workCount_ptr;

    int i = lv-1;
    uint sol = 0;
    int my = 0;
    uint mid = 0;
    uint diag1 = 0, diag1r = 0;
    uint diag2 = 0, diag2r = 0;
    if (progress[gid].i == 87) {
        i = lv-1;
        my = atomic_fetch_add_explicit(flag, 1, metal::memory_order_relaxed);
        if (my < workCount) {
            mid = works[my].mid;
            diag1 = works[my].diag1, diag1r = 0;
            diag2 = works[my].diag2, diag2r = 0;
            s0[i][tid] = canplace[i] & ~(mid | diag1 | diag2);
        }
        else {
            i = 87;
        }
    }
    else {
        my = -1;
        i = progress[gid].i;
        mid = progress[gid].mid;
        diag1 = progress[gid].diag1;
        diag1r = progress[gid].diag1r;
        diag2 = progress[gid].diag2;
        diag2r = progress[gid].diag2r;
        for (int j = 0; j < 12; j++) {
            s0[j][tid] = progress[gid].s0[j];
        }
    }
    uint runtime = 0;
    for (; runtime < 100000 && my < workCount; ) {
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
            else {
                i = 87;
            }
        }
        runtime++;
    }
    result[gid] = sol;

    progress[gid].i = i;
    progress[gid].mid = mid;
    progress[gid].diag1 = diag1;
    progress[gid].diag1r = diag1r;
    progress[gid].diag2 = diag2;
    progress[gid].diag2r = diag2r;
    for (int j = 0; j < 12; j++) {
        progress[gid].s0[j] = s0[j][tid];
    }
}
