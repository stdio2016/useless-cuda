#include <immintrin.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>

#ifdef _MSC_VER
#define POPCNT __popcnt
#else
#define POPCNT __builtin_popcount
#endif

typedef unsigned int uint;
class Timing {
public:
  // automatically record start time
  Timing();
  
  // get time in milliseconds
  double getRunTime(bool cont=false);

private:
  std::chrono::steady_clock::time_point startTime;
};

Timing::Timing() {
  startTime = std::chrono::steady_clock::now();
}

double Timing::getRunTime(bool cont) {
  auto now = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> t = now - startTime;
  if (!cont) {
    startTime = now;
  }
  return t.count();
}

struct SubProbs {
  std::vector<uint> mid, diag1, diag2, choice;
  int cnt;
  SubProbs(int n): mid(n), diag1(n), diag2(n), choice(n), cnt(0) {}
};

__m256i *build_shuffle_table() {
  __m256i *shuffle_table = (__m256i*)_mm_malloc(256 * sizeof(__m256), sizeof(__m256));
  for (int i = 0; i < 256; i++) {
    int pp[8] = {0};
    int cnt = 0;
    for (int j = 0; j < 8; j++) {
      if (!(i>>j&1)) pp[cnt++] = j;
    }
    shuffle_table[i] = _mm256_loadu_si256((__m256i*)pp);
  }
  return shuffle_table;
}

int avx2_compaction(int n, uint *mask, uint *dat, const __m256i *shuffle_table) {
  int i;
  int cnt = 0;
  for (i = 0; i <= n-8; i += 8) {
    __m256i m = _mm256_loadu_si256((__m256i*)&mask[i]);
    __m256i d = _mm256_loadu_si256((__m256i*)&dat[i]);
    __m256i zer = _mm256_cmpeq_epi32(m, _mm256_setzero_si256());
    int sel = _mm256_movemask_ps(_mm256_castsi256_ps(zer));
    __m256i idx = _mm256_load_si256(&shuffle_table[sel]);
    __m256i out = _mm256_permutevar8x32_epi32(d, idx);
    _mm256_storeu_si256((__m256i*)&dat[cnt], out);
    cnt += POPCNT(255-sel);
  }
  for (; i < n; i++) {
    if (mask[i]) {
      dat[cnt] = dat[i];
      cnt += 1;
    }
  }
  return cnt;
}

int compaction(int n, uint *mask, uint *dat) {
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    if (mask[i]) {
      dat[cnt] = dat[i];
      cnt += 1;
    }
  }
  return cnt;
}

int next_step(SubProbs &a, SubProbs &b, int from, int to, uint canplace, const __m256i *shuffle_table) {
  int off = b.cnt;
  int i;
  int rem = from;
  __m256i can = _mm256_set1_epi32(canplace);
  for (i = from; i+8 <= to; i += 8) {
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
    int sel = _mm256_movemask_ps(_mm256_castsi256_ps(zer));
    __m256i idx = _mm256_load_si256(&shuffle_table[sel]);
    nxt_cho = _mm256_permutevar8x32_epi32(nxt_cho, idx);
    nxt_mid = _mm256_permutevar8x32_epi32(nxt_mid, idx);
    nxt_d1 = _mm256_permutevar8x32_epi32(nxt_d1, idx);
    nxt_d2 = _mm256_permutevar8x32_epi32(nxt_d2, idx);
    
    _mm256_storeu_si256((__m256i*)&b.choice[off], nxt_cho);
    _mm256_storeu_si256((__m256i*)&b.mid[off], nxt_mid);
    _mm256_storeu_si256((__m256i*)&b.diag1[off], nxt_d1);
    _mm256_storeu_si256((__m256i*)&b.diag2[off], nxt_d2);
    off += POPCNT(255-sel);
    
    zer = _mm256_cmpeq_epi32(cho, _mm256_setzero_si256());
    sel = _mm256_movemask_ps(_mm256_castsi256_ps(zer));
    idx = _mm256_load_si256(&shuffle_table[sel]);
    cho = _mm256_permutevar8x32_epi32(cho, idx);
    mid = _mm256_permutevar8x32_epi32(mid, idx);
    d1 = _mm256_permutevar8x32_epi32(d1, idx);
    d2 = _mm256_permutevar8x32_epi32(d2, idx);
    
    _mm256_storeu_si256((__m256i*)&a.choice[rem], cho);
    _mm256_storeu_si256((__m256i*)&a.mid[rem], mid);
    _mm256_storeu_si256((__m256i*)&a.diag1[rem], d1);
    _mm256_storeu_si256((__m256i*)&a.diag2[rem], d2);
    rem += POPCNT(255-sel);
  }
  for (; i < to; i++) {
    uint cho = a.choice[i];
    uint mid = a.mid[i];
    uint d1 = a.diag1[i];
    uint d2 = a.diag2[i];
    uint lowbit = cho & -cho;
    uint nxt_mid = mid | lowbit;
    uint nxt_d1 = (d1 | lowbit) << 1;
    uint nxt_d2 = (d2 | lowbit) >> 1;
    cho = cho - lowbit;
    uint nxt_cho = canplace & ~(nxt_mid | nxt_d1 | nxt_d2);
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

int solve(int n, const uint *mask, SubProbs in, int page, const __m256i *table) {
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
    int all = probs[lv].cnt;
    if (page < has) has = page;
    int prev = probs[lv+1].cnt;
    int rem = next_step(probs[lv], probs[lv+1], all - has, all, mask[lv+1], table);
    probs[lv].cnt = rem;
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

long long nqueen_avx2(int n, const uint *mask, const __m256i *shuffle_table) {
  SubProbs gen(1);
  gen.cnt = 1;
  gen.choice[0] = mask[0];
  gen.mid[0] = 0;
  gen.diag1[0] = 0;
  gen.diag2[0] = 0;
  return solve(n, mask, gen, 256, shuffle_table);
}

long long nqueen_parallel_avx2(int n, const uint *mask, const __m256i *shuffle_table) {
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
      rem = next_step(gen, gen2, 0, gen.cnt, mask[unroll_lv], shuffle_table);
      gen.cnt = rem;
    }
    std::swap(gen, gen2);
  }
  if (unroll_lv >= n-7) {
    // too little remaining works
    return solve(n-unroll_lv, &mask[unroll_lv], gen, 256, shuffle_table);;
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
    
    ans += solve(n-unroll_lv, &mask[unroll_lv], me, 256, shuffle_table);
    //printf("ans = %d\n", ans);
  }
  return ans;
}

int main(int argc, char *argv[]) {
  __m256i *shuffle_table = build_shuffle_table();
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
    
    std::vector<uint> mask(n);
    for (int i = 0; i < n; i++) {
      fgets(buf, 100, filein);
      mask[i] = (1u<<n)-1;
      for (int j = 0; j < n; j++) {
        if (buf[j] == '*') mask[i] -= 1u<<j;
      }
    }
    Timing tm;
    long long ans = nqueen_parallel_avx2(n, mask.data(), shuffle_table);
    double t1 = tm.getRunTime();
    printf("Case #%d: %lld\n", T, ans);
  }
}
