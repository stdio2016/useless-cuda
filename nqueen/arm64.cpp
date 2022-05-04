#include<stdio.h>
#include<arm_neon.h>
#include<chrono>
#include<vector>
#include<cstdint>
using namespace std::chrono;

__attribute((aligned(16)))
static const uint8_t compressShuffle[16][16] = {
    { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 0, 1, 2, 3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 4, 5, 6, 7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 8, 9, 10, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 0, 1, 2, 3, 8, 9, 10, 11, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 4, 5, 6, 7, 8, 9, 10, 11, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 255, 255, 255 },
    { 12, 13, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 0, 1, 2, 3, 12, 13, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 4, 5, 6, 7, 12, 13, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 255, 255, 255, 255 },
    { 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 255, 255, 255, 255, 255, 255 },
    { 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 255, 255 },
    { 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 255, 255 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }
};

__attribute((aligned(16)))
static const uint8_t gather_data[16] = {12, 8, 4, 0, 12, 8, 4, 0, 28, 24, 20, 16, 28, 24, 20, 16};

__attribute((aligned(16)))
static const uint32_t aggregate_data[4] = {0x8040201, 0x1010101, 0x8040201, 0x1010101};

struct SubProbs {
  std::vector<uint32_t> mid, diag1, diag2, choice;
  int cnt;
  SubProbs(int n): mid(n), diag1(n), diag2(n), choice(n), cnt(0) {}
};

// reference https://github.com/lemire/streamvbyte
int compress(int *a, int n) {
    uint8x8_t gather = vld1_u8(gather_data);
    uint8x16_t gather2 = vld1q_u8(gather_data);
    uint32x2_t aggregate = vld1_u32(aggregate_data);
    uint32x4_t aggregate2 = vld1q_u32(aggregate_data);
    int j = 0;
    for (int i = 0; i < n; i += 8) {
        int32x4x2_t x = vld2q_s32(&a[i]);
        uint32x4_t mask1 = vceqzq_u32(x.val[0]);
        uint32x4_t mask2 = vceqzq_u32(x.val[1]);
        // mask[i] is 0 if x[i] == 0, 1 otherwise
        mask1 = vaddq_u32(vdupq_n_u32(1), mask1);
        mask2 = vaddq_u32(vdupq_n_u32(1), mask2);

        /*uint8x16_t mask1_b = vreinterpretq_u8_u32(mask1);
        uint8x16_t mask2_b = vreinterpretq_u8_u32(mask2);
        uint8x8_t lobytes1 = vqtbl1_u8(mask1_b, gather);
        uint8x8_t lobytes2 = vqtbl1_u8(mask2_b, gather);
        uint32x2_t res1 = vmul_u32(lobytes1, aggregate);
        uint32x2_t res2 = vmul_u32(lobytes2, aggregate);
        lobytes1 = vreinterpret_u8_u32(res1);
        lobytes2 = vreinterpret_u8_u32(res2);
        unsigned code1 = lobytes1[3];
        unsigned len1 = lobytes1[7];
        unsigned code2 = lobytes2[3];
        unsigned len2 = lobytes2[7];*/
        
        uint8x16x2_t mask_b = {vreinterpretq_u8_u32(mask1), vreinterpretq_u8_u32(mask2)};
        uint8x16_t lobytes = vqtbl2q_u8(mask_b, gather2);
        uint32x4_t res = vmulq_u32(lobytes, aggregate2);
        lobytes = vreinterpretq_u8_u32(res);

        unsigned code1 = lobytes[3];
        unsigned len1 = lobytes[7];
        unsigned code2 = lobytes[11];
        unsigned len2 = lobytes[15];
        //printf("code %d len %u\n", code, len);
        uint8x16_t y1 = vreinterpretq_u8_s32(x.val[0]);
        y1 = vqtbl1q_u8(y1, vld1q_u8(compressShuffle[code1]));
        uint8x16_t y2 = vreinterpretq_u8_s32(x.val[1]);
        y2 = vqtbl1q_u8(y2, vld1q_u8(compressShuffle[code2]));
        vst1q_s32(&a[j], vreinterpretq_s32_u8(y1));
        j += len1;
        vst1q_s32(&a[j], vreinterpretq_s32_u8(y2));
        j += len2;
    }
    return j;
}

int compress2(int *a, int n) {
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] != 0) {
            a[j++] = a[i];
        }
    }
    return j;
}

// reference https://github.com/lemire/streamvbyte
inline void compress_shuffle_idx(
        uint32x4_t x1, uint32x4_t x2,
        unsigned &len1, unsigned &len2,
        uint8x16_t &idx1, uint8x16_t &idx2)
{
    uint8x16_t gather2 = vld1q_u8(gather_data);
    uint32x4_t aggregate2 = vld1q_u32(aggregate_data);
    uint32x4_t mask1 = vceqzq_u32(x1);
    uint32x4_t mask2 = vceqzq_u32(x2);
    // mask[i] is 0 if x[i] == 0, 1 otherwise
    mask1 = vaddq_u32(vdupq_n_u32(1), mask1);
    mask2 = vaddq_u32(vdupq_n_u32(1), mask2);
    
    uint8x16x2_t mask_b = {vreinterpretq_u8_u32(mask1), vreinterpretq_u8_u32(mask2)};
    uint8x16_t lobytes = vqtbl2q_u8(mask_b, gather2);
    uint32x4_t res = vmulq_u32(lobytes, aggregate2);
    uint32_t codelen[4];
    vst1q_u32(codelen, res);
    //lobytes = vreinterpretq_u8_u32(res);

    //unsigned code1 = lobytes[3];
    unsigned code1 = codelen[0] >> 20;
    //unsigned code2 = lobytes[11];
    unsigned code2 = codelen[2] >> 20;
    //len1 = lobytes[7];
    len1 = codelen[1] >> 24;
    //len2 = lobytes[15];
    len2 = codelen[3] >> 24;
    //printf("code %d len %u\n", code, len);
    idx1 = vld1q_u8(&compressShuffle[0][0] + code1);
    idx2 = vld1q_u8(&compressShuffle[0][0] + code2);
}

inline uint32x4_t getlowbit(uint32x4_t x) {
    return vandq_u32(
        x,
        vreinterpretq_u32_s32(vnegq_s32(vreinterpretq_s32_u32(x)))
    );
}

inline uint32x4_t lookup_u32(uint32x4_t x, uint8x16_t shuffle) {
    return vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), shuffle));
}

int next_step(SubProbs &a, SubProbs &b, int from, int to, uint32_t canplace) {
    int off = b.cnt;
    int i = from;
    int rem = from;
    uint32x4_t can = vld1q_dup_u32(&canplace);
    uint32_t *a_choice = &a.choice[0];
    uint32_t *a_mid = &a.mid[0];
    uint32_t *a_diag1 = &a.diag1[0];
    uint32_t *a_diag2 = &a.diag2[0];
    uint32_t *b_choice = &b.choice[0];
    uint32_t *b_mid = &b.mid[0];
    uint32_t *b_diag1 = &b.diag1[0];
    uint32_t *b_diag2 = &b.diag2[0];
    for (i = from; i+8 <= to; i += 8) {
        uint32x4_t cho_1 = vld1q_u32(&a_choice[i]);
        uint32x4_t cho_2 = vld1q_u32(&a_choice[i+4]);
        uint32x4_t mid_1 = vld1q_u32(&a_mid[i]);
        uint32x4_t mid_2 = vld1q_u32(&a_mid[i+4]);
        uint32x4_t d1_1 = vld1q_u32(&a_diag1[i]);
        uint32x4_t d1_2 = vld1q_u32(&a_diag1[i+4]);
        uint32x4_t d2_1 = vld1q_u32(&a_diag2[i]);
        uint32x4_t d2_2 = vld1q_u32(&a_diag2[i+4]);
        uint32x4_t lowbit_1 = getlowbit(cho_1);
        uint32x4_t lowbit_2 = getlowbit(cho_2);
        uint32x4_t nxt_mid_1 = vorrq_u32(mid_1, lowbit_1);
        uint32x4_t nxt_mid_2 = vorrq_u32(mid_2, lowbit_2);
        uint32x4_t nxt_d1_1 = vshlq_n_u32(vorrq_u32(d1_1, lowbit_1), 1);
        uint32x4_t nxt_d1_2 = vshlq_n_u32(vorrq_u32(d1_2, lowbit_2), 1);
        uint32x4_t nxt_d2_1 = vshrq_n_u32(vorrq_u32(d2_1, lowbit_1), 1);
        uint32x4_t nxt_d2_2 = vshrq_n_u32(vorrq_u32(d2_2, lowbit_2), 1);
        cho_1 = vsubq_u32(cho_1, lowbit_1);
        cho_2 = vsubq_u32(cho_2, lowbit_2);
        // bic(a, b) means a AND NOT b
        uint32x4_t nxt_cho_1 = vbicq_u32(can, vorrq_u32(vorrq_u32(nxt_mid_1, nxt_d1_1), nxt_d2_1));
        uint32x4_t nxt_cho_2 = vbicq_u32(can, vorrq_u32(vorrq_u32(nxt_mid_2, nxt_d1_2), nxt_d2_2));

        unsigned len3, len4;
        uint8x16_t idx3, idx4;
        compress_shuffle_idx(cho_1, cho_2, len3, len4, idx3, idx4);
        cho_1 = lookup_u32(cho_1, idx3);
        cho_2 = lookup_u32(cho_2, idx4);
        mid_1 = lookup_u32(mid_1, idx3);
        mid_2 = lookup_u32(mid_2, idx4);
        d1_1 = lookup_u32(d1_1, idx3);
        d1_2 = lookup_u32(d1_2, idx4);
        d2_1 = lookup_u32(d2_1, idx3);
        d2_2 = lookup_u32(d2_2, idx4);

        unsigned len1, len2;
        uint8x16_t idx1, idx2;
        compress_shuffle_idx(nxt_cho_1, nxt_cho_2, len1, len2, idx1, idx2);
        nxt_cho_1 = lookup_u32(nxt_cho_1, idx1);
        nxt_cho_2 = lookup_u32(nxt_cho_2, idx2);
        nxt_mid_1 = lookup_u32(nxt_mid_1, idx1);
        nxt_mid_2 = lookup_u32(nxt_mid_2, idx2);
        nxt_d1_1 = lookup_u32(nxt_d1_1, idx1);
        nxt_d1_2 = lookup_u32(nxt_d1_2, idx2);
        nxt_d2_1 = lookup_u32(nxt_d2_1, idx1);
        nxt_d2_2 = lookup_u32(nxt_d2_2, idx2);

        vst1q_u32(&a_choice[rem], cho_1);
        vst1q_u32(&a_mid[rem], mid_1);
        vst1q_u32(&a_diag1[rem], d1_1);
        vst1q_u32(&a_diag2[rem], d2_1);
        rem += len3;
        vst1q_u32(&a_choice[rem], cho_2);
        vst1q_u32(&a_mid[rem], mid_2);
        vst1q_u32(&a_diag1[rem], d1_2);
        vst1q_u32(&a_diag2[rem], d2_2);
        rem += len4;

        vst1q_u32(&b_choice[off], nxt_cho_1);
        vst1q_u32(&b_mid[off], nxt_mid_1);
        vst1q_u32(&b_diag1[off], nxt_d1_1);
        vst1q_u32(&b_diag2[off], nxt_d2_1);
        off += len1;
        vst1q_u32(&b_choice[off], nxt_cho_2);
        vst1q_u32(&b_mid[off], nxt_mid_2);
        vst1q_u32(&b_diag1[off], nxt_d1_2);
        vst1q_u32(&b_diag2[off], nxt_d2_2);
        off += len2;
    }
    for (; i < to; i++) {
        uint32_t cho = a_choice[i];
        uint32_t mid = a_mid[i];
        uint32_t d1 = a_diag1[i];
        uint32_t d2 = a_diag2[i];
        uint32_t lowbit = cho & -cho;
        uint32_t nxt_mid = mid | lowbit;
        uint32_t nxt_d1 = (d1 | lowbit) << 1;
        uint32_t nxt_d2 = (d2 | lowbit) >> 1;
        cho = cho - lowbit;
        uint32_t nxt_cho = canplace & ~(nxt_mid | nxt_d1 | nxt_d2);
        if (cho) {
            a_choice[rem] = cho;
            a_mid[rem] = mid;
            a_diag1[rem] = d1;
            a_diag2[rem] = d2;
            rem++;
        }
        if (nxt_cho) {
            b_choice[off] = nxt_cho;
            b_mid[off] = nxt_mid;
            b_diag1[off] = nxt_d1;
            b_diag2[off] = nxt_d2;
            off++;
        }
    }
    b.cnt = off;
    return rem;
}

long long solve(int n, const uint32_t *mask, SubProbs in, int page) {
    std::vector<SubProbs> probs;
    probs.push_back(in);
    for (int i = 1; i <= n; i++) {
        probs.push_back(SubProbs(page * 2));
    }
    long long ans = 0;
    int lv = 0;
    int depleted = 0;
    while (depleted < n-1) {
        int has = probs[lv].cnt;
        int all = probs[lv].cnt;
        if (page < has) has = page;
        int prev = probs[lv+1].cnt;
        int rem = next_step(probs[lv], probs[lv+1], all - has, all, mask[lv+1]);
        /*if (all > has) {
            std::copy(&probs[lv].mid[has], &probs[lv].mid[all], &probs[lv].mid[rem]);
            std::copy(&probs[lv].diag1[has], &probs[lv].diag1[all], &probs[lv].diag1[rem]);
            std::copy(&probs[lv].diag2[has], &probs[lv].diag2[all], &probs[lv].diag2[rem]);
            std::copy(&probs[lv].choice[has], &probs[lv].choice[all], &probs[lv].choice[rem]);
        }
        probs[lv].cnt = rem + all - has;*/
        probs[lv].cnt = rem;
        if (probs[lv].cnt == 0 && lv == depleted) depleted++;
        ans += has;

        //printf("lv=%d added=%d rem=%d depleted=%d cnt=%d,%d\n", lv, now-prev, rem, depleted, probs[lv].cnt, probs[lv+1].cnt);

        if (probs[lv].cnt < page && lv > depleted) lv -= 1;
        else while (probs[lv+1].cnt >= page || lv+1 == depleted) {
            lv += 1;
            if (lv == n-1) {
                //ans += probs[n-1].cnt;
                probs[n-1].cnt = 0;
                lv -= 1;
                break;
            }
        }
    }
    return ans;
}

long long nqueen_arm64(int n, const uint32_t *mask) {
    SubProbs gen(1);
    gen.cnt = 1;
    gen.choice[0] = mask[0];
    gen.mid[0] = 0;
    gen.diag1[0] = 0;
    gen.diag2[0] = 0;
    return solve(n, mask, gen, 256);
}

long long nqueen_parallel_arm64(int n, const uint32_t *mask) {
    if (n == 1) { // boundary/trivial case
        return (mask[0]&1) == 1;
    }
    SubProbs gen(1);
    gen.cnt = 1;
    gen.choice[0] = mask[0];
    gen.mid[0] = 0;
    gen.diag1[0] = 0;
    gen.diag2[0] = 0;
    int unroll_lv = 0;
    while (unroll_lv < n-7 && gen.cnt < 1000) {
        unroll_lv += 1;
        SubProbs gen2(gen.cnt * n);
        gen2.cnt = 0;
        int rem = gen.cnt;
        while (rem > 0) {
            rem = next_step(gen, gen2, 0, gen.cnt, mask[unroll_lv]);
            gen.cnt = rem;
        }
        std::swap(gen, gen2);
    }
    if (unroll_lv >= n-7) {
        // too little remaining works
        return solve(n-unroll_lv, &mask[unroll_lv], gen, 256);;
    }

    long long ans = 0;
    #pragma omp parallel reduction(+:ans)
    {
        SubProbs me(0);
        me.cnt = 0;
        // do not change schedule
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < gen.cnt; i++) {
            me.cnt += 1;
            me.choice.push_back(gen.choice[i]);
            me.mid.push_back(gen.mid[i]);
            me.diag1.push_back(gen.diag1[i]);
            me.diag2.push_back(gen.diag2[i]);
        }
        
        ans += solve(n-unroll_lv, &mask[unroll_lv], me, 384);
        //printf("ans = %d\n", ans);
    }
    return ans;
}

int main(int argc, char *argv[]) {
    int n = 0;
    int page = 256;
    char buf[100];
    FILE *filein = stdin;
    int T = 0;
    for (int i = 1; i < argc; i++) {
        if (i+1<argc && filein == stdin && strcmp(argv[i], "-i") == 0) {
            filein = fopen(argv[i+1], "r");
            if (filein == NULL) {
                fprintf(stderr, "cannot open file\n");
                return 1;
            }
        }
    }
    while (fscanf(filein, "%d", &n) == 1) {
        fgets(buf, 100, filein);
        T += 1;
        if (n < 1 || n >= 32) return 0;
        
        std::vector<uint32_t> mask(n);
        for (int i = 0; i < n; i++) {
            fgets(buf, 100, filein);
            mask[i] = (1u<<n)-1;
            for (int j = 0; j < n; j++) {
                if (buf[j] == '*') mask[i] -= 1u<<j;
            }
        }
        //Timing tm;
        long long ans = nqueen_parallel_arm64(n, mask.data());
        //double t1 = tm.getRunTime();
        printf("Case #%d: %lld\n", T, ans);
    }
}
