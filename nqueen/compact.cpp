#include<stdio.h>
#include<arm_neon.h>
#include<chrono>
#include<vector>
#include<cstdint>
using namespace std::chrono;

const uint8_t compressShuffle[16][16] = {
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

const uint8_t gather_data[16] = {12, 8, 4, 0, 12, 8, 4, 0, 28, 24, 20, 16, 28, 24, 20, 16};

const uint32_t aggregate_data[4] = {0x8040201, 0x1010101, 0x8040201, 0x1010101};

struct SubProbs {
  std::vector<uint32_t> mid, diag1, diag2, choice;
  int cnt;
  SubProbs(int n): mid(n), diag1(n), diag2(n), choice(n), cnt(0) {}
};

int compress(int *a, int n) {
    //some(__builtin_neon_vld1q_v(gather_data, 48));
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
int next_step(SubProbs &a, SubProbs &b, int num, uint32_t canplace) {
    int off = b.cnt;
    int i = 0;
    int rem = 0;
    /*__m256i can = _mm256_set1_epi32(canplace);
    for (i = 0; i+8 <= num; i += 8) {
        __m256i cho = _mm256_loadu_si256((__m256i*)&a.choice[i]);
        __m256i mid = _mm256_loadu_si256((__m256i*)&a.mid[i]);
        __m256i d1 = _mm256_loadu_si256((__m256i*)&a.diag1[i]);
        __m256i d2 = _mm256_loadu_si256((__m256i*)&a.diag2[i]);
        __m256i lowbit = _mm256_and_si256(cho, _mm256_sub_epi32(_mm256_setzero_si256(), cho));
        __m256i nxt_mid = _mm256_or_si256(mid, lowbit);
        __m256i nxt_d1 = _mm256_slli_epi32(_mm256_or_si256(d1, lowbit), 1);
        __m256i nxt_d2 = _mm256_srli_epi32(_mm256_or_si256(d2, lowbit), 1);
        cho = _mm256_sub_epi32(cho, lowbit);
        __m256i nxt_cho = _mm256_andnot_si256(_mm256_or_si256(_mm256_or_si256(nxt_mid, nxt_d1), nxt_d2), can);
        
        __m256i zer = _mm256_cmpeq_epi32(nxt_cho, _mm256_setzero_si256());
        int sel = _mm256_movemask_ps((__m256)zer);
        __m256i idx = _mm256_load_si256(&shuffle_table[sel]);
        nxt_cho = (__m256i)_mm256_permutevar8x32_ps((__m256)nxt_cho, idx);
        nxt_mid = (__m256i)_mm256_permutevar8x32_ps((__m256)nxt_mid, idx);
        nxt_d1 = (__m256i)_mm256_permutevar8x32_ps((__m256)nxt_d1, idx);
        nxt_d2 = (__m256i)_mm256_permutevar8x32_ps((__m256)nxt_d2, idx);
        
        _mm256_storeu_si256((__m256i*)&b.choice[off], nxt_cho);
        _mm256_storeu_si256((__m256i*)&b.mid[off], nxt_mid);
        _mm256_storeu_si256((__m256i*)&b.diag1[off], nxt_d1);
        _mm256_storeu_si256((__m256i*)&b.diag2[off], nxt_d2);
        off += __builtin_popcount(255-sel);
        
        zer = _mm256_cmpeq_epi32(cho, _mm256_setzero_si256());
        sel = _mm256_movemask_ps((__m256)zer);
        idx = _mm256_load_si256(&shuffle_table[sel]);
        cho = (__m256i)_mm256_permutevar8x32_ps((__m256)cho, idx);
        mid = (__m256i)_mm256_permutevar8x32_ps((__m256)mid, idx);
        d1 = (__m256i)_mm256_permutevar8x32_ps((__m256)d1, idx);
        d2 = (__m256i)_mm256_permutevar8x32_ps((__m256)d2, idx);
        
        _mm256_storeu_si256((__m256i*)&a.choice[rem], cho);
        _mm256_storeu_si256((__m256i*)&a.mid[rem], mid);
        _mm256_storeu_si256((__m256i*)&a.diag1[rem], d1);
        _mm256_storeu_si256((__m256i*)&a.diag2[rem], d2);
        rem += __builtin_popcount(255-sel);
    }*/
    for (; i < num; i++) {
        uint32_t cho = a.choice[i];
        uint32_t mid = a.mid[i];
        uint32_t d1 = a.diag1[i];
        uint32_t d2 = a.diag2[i];
        uint32_t lowbit = cho & -cho;
        uint32_t nxt_mid = mid | lowbit;
        uint32_t nxt_d1 = (d1 | lowbit) << 1;
        uint32_t nxt_d2 = (d2 | lowbit) >> 1;
        cho = cho - lowbit;
        uint32_t nxt_cho = canplace & ~(nxt_mid | nxt_d1 | nxt_d2);
        if (cho) {
            a.choice[rem] = cho;
            a.mid[rem] = mid;
            a.diag1[rem] = d1;
            a.diag2[rem] = d2;
            rem++;
        }
        if (nxt_cho) {
            b.choice[off] = nxt_cho;
            b.mid[off] = nxt_mid;
            b.diag1[off] = nxt_d1;
            b.diag2[off] = nxt_d2;
            off++;
        }
    }
    b.cnt = off;
    return rem;
}

int solve(int n, const uint32_t *mask, SubProbs in, int page) {
    std::vector<SubProbs> probs;
    probs.push_back(in);
    for (int i = 1; i <= n; i++) {
        probs.push_back(SubProbs(page * 2));
    }
    int ans = 0;
    int lv = 0;
    int depleted = 0;
    while (depleted < n-1) {
        int has = probs[lv].cnt;
        if (page < has) has = page;
        int prev = probs[lv+1].cnt;
        int rem = next_step(probs[lv], probs[lv+1], has, mask[lv+1]);
        int all = probs[lv].cnt;
        if (all > has) {
            std::copy(&probs[lv].mid[has], &probs[lv].mid[all], &probs[lv].mid[rem]);
            std::copy(&probs[lv].diag1[has], &probs[lv].diag1[all], &probs[lv].diag1[rem]);
            std::copy(&probs[lv].diag2[has], &probs[lv].diag2[all], &probs[lv].diag2[rem]);
            std::copy(&probs[lv].choice[has], &probs[lv].choice[all], &probs[lv].choice[rem]);
        }
        probs[lv].cnt = rem + all - has;
        if (probs[lv].cnt == 0 && lv == depleted) depleted++;

        //printf("lv=%d added=%d rem=%d depleted=%d cnt=%d,%d\n", lv, now-prev, rem, depleted, probs[lv].cnt, probs[lv+1].cnt);

        if (probs[lv].cnt < page && lv > depleted) lv -= 1;
        else while (probs[lv+1].cnt >= page || lv+1 == depleted) {
            lv += 1;
            if (lv == n-1) {
                ans += probs[n-1].cnt;
                probs[n-1].cnt = 0;
                lv -= 1;
                break;
            }
        }
    }
    return ans;
}

long long nqueen_avx2(int n, const uint32_t *mask) {
    SubProbs gen(1);
    gen.cnt = 1;
    gen.choice[0] = mask[0];
    gen.mid[0] = 0;
    gen.diag1[0] = 0;
    gen.diag2[0] = 0;
    return solve(n, mask, gen, 256);
}

long long nqueen_parallel_avx2(int n, const uint32_t *mask) {
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
            rem = next_step(gen, gen2, gen.cnt, mask[unroll_lv]);
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
        
        ans += solve(n-unroll_lv, &mask[unroll_lv], me, 256);
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
        long long ans = nqueen_parallel_avx2(n, mask.data());
        //double t1 = tm.getRunTime();
        printf("Case #%d: %lld\n", T, ans);
    }
}
