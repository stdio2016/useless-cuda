#include <immintrin.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>

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
  __m256i *shuffle_table = (__m256i*)_aligned_malloc(256 * sizeof(__m256), sizeof(__m256));
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
    int sel = _mm256_movemask_ps((__m256)zer);
    __m256i idx = _mm256_load_si256(&shuffle_table[sel]);
    __m256i out = (__m256i)_mm256_permutevar8x32_ps((__m256)d, idx);
    _mm256_storeu_si256((__m256i*)&dat[cnt], out);
    cnt += __builtin_popcount(255-sel);
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

int next_step(SubProbs &a, SubProbs &b, int num, uint canplace, const __m256i *shuffle_table) {
  int off = b.cnt;
  int i;
  int rem = 0;
  __m256i can = _mm256_set1_epi32(canplace);
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
  }
  for (; i < num; i++) {
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

int main() {
  __m256i *table = build_shuffle_table();
  int n = 0;
  int page = 256;
  scanf("%d", &n);
  std::vector<SubProbs> probs;
  for (int i = 0; i <= n; i++) {
    probs.push_back(SubProbs(page * 2));
  }
  probs[0].cnt = 1;
  probs[0].choice[0] = (1<<n)-1;
  probs[0].mid[0] = 0;
  probs[0].diag1[0] = 0;
  probs[0].diag2[0] = 0;
  Timing tm;
  
  tm.getRunTime();
  int ans = 0;
  int lv = 0;
  int depleted = 0;
  while (depleted < n-1) {
    int has = probs[lv].cnt;
    if (page < has) has = page;
    int prev = probs[lv+1].cnt;
    int rem = next_step(probs[lv], probs[lv+1], has, (1<<n)-1, table);
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
  double t1 = tm.getRunTime();
  
  printf("time=%f ans=%d\n", t1, ans);
}
