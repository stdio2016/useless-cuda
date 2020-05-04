/**
 * Ideas from:
 * https://medium.com/fcamels-notes/%E4%BD%BF%E7%94%A8%E4%BD%8D%E5%85%83%E8%A8%88%E7%AE%97%E5%8A%A0%E9%80%9F%E6%B1%82%E8%A7%A3-n-queen-d20442f34110
 * https://morris821028.github.io/2016/04/30/jg-10026/
 * Optimization of N-Queens Solvers on Graphics Processors. Tao Zhang, Wei Shu,
   Min-You Wu. In: Proceedings of the 9th International Conference on Advanced
   Parallel Processing Technologies 142-156
**/
#include<stdio.h>
#include<cstring>

typedef unsigned uint;
uint canplace[32];

// use recursive method (bit mask)
int nqueen_recur(int lv, uint mid, uint diag1, uint diag2) {
  if (lv == 0) return 1;
  int count = 0;
  uint choice = canplace[lv-1] & ~(mid | diag1 | diag2);
  while (choice) {
    uint i = choice & -choice;
    choice -= i;
    count += nqueen_recur(lv-1, mid|i, (diag1|i)<<1, (diag2|i)>>1);
  }
  return count;
}

// turn recursion into iteration
int nqueen_iter(int lv, uint mid, uint diag1, uint diag2, uint mask) {
  uint s0[32], s1[32], s2[32], s3[32], s4[32];
  if (lv == 0) return 1;
  int count = 0, n = lv-1, i = lv-1;
  uint choice = canplace[lv-1] & ~(mid | diag1 | diag2);
  s0[i] = choice;
  s1[i] = mid;
  s2[i] = diag1;
  s3[i] = diag2;
  s4[i] = n;
  while (i < lv) {
    choice = s0[i];
    mid = s1[i];
    diag1 = s2[i];
    diag2 = s3[i];
    uint bit = choice & -choice;
    s0[i] = choice - bit;
    if (mid + bit == mask) {
      count += 1;
    }
    else {
      mid = mid + bit;
      diag1 = (diag1 | bit) << 1;
      diag2 = (diag2 | bit) >> 1;
      n -= 1;
      i -= choice != bit;
      choice = canplace[n] & ~(mid | diag1 | diag2);
      s0[i] = choice;
      s1[i] = mid;
      s2[i] = diag1;
      s3[i] = diag2;
      s4[i] = n;
      if (choice == 0) n = s4[i+1], i += 1;
    }
  }
  return count;
}

struct SubP {
  uint mid, diag1, diag2;  
} SP[10000*32];
int nprobs;

// generate subproblems
void nqueen_gen(int lv, uint mid, uint diag1, uint diag2, int cut) {
  if (lv == cut) {
    SubP p = {mid, diag1, diag2};
    SP[nprobs++] = p;
    return;
  }
  uint choice = canplace[lv-1] & ~(mid | diag1 | diag2);
  while (choice) {
    uint i = choice & -choice;
    choice -= i;
    nqueen_gen(lv-1, mid|i, (diag1|i)<<1, (diag2|i)>>1, cut);
  }
}

int nqueen_parallel(int n, int method) {
  // create subproblems
  int cut;
  for (cut = n-1; cut >= 5; cut--) {
    nprobs = 0;
    nqueen_gen(n, 0, 0, 0, cut);
    if (nprobs >= 10000) break;
  }

  // too little subproblems
  if (cut < 5) cut = 5;

  // n is too small
  if (n <= 5) {
    nprobs = 1;
    cut = n;
    SP[0].mid = 0;
    SP[0].diag1 = 0;
    SP[0].diag2 = 0;
  }

  int ans = 0;
  #pragma omp parallel for reduction(+:ans) schedule(static, 100)
  for (int i = 0; i < nprobs; i++) {
    SubP p = SP[i];
    if (method == 0) ans += nqueen_recur(cut, p.mid, p.diag1, p.diag2);
    else if (method == 1) ans += nqueen_iter(cut, p.mid, p.diag1, p.diag2, (1u<<n)-1);
  }
  return ans;
}

int main(int argc, char *argv[]) {
  int n;
  int T = 0;
  char buf[100];
  int method = 0, parallel = 0;
  FILE *filein = stdin;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-f") == 0) method = 1;
    if (strcmp(argv[i], "-p") == 0) parallel = 1;
    else if (i+1<argc && filein == stdin && strcmp(argv[i], "-i") == 0) {
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
    for (int i = 0; i < n; i++) {
      fgets(buf, 100, filein);
      canplace[n-1-i] = (1u<<n)-1;
      for (int j = 0; j < n; j++) {
        if (buf[j] == '*') canplace[n-1-i] -= 1u<<j;
      }
    }

    int ans = 0;
    if (parallel) ans = nqueen_parallel(n, method);
    else if (method == 0) ans = nqueen_recur(n, 0, 0, 0);
    else if (method == 1) ans = nqueen_iter(n, 0, 0, 0, (1u<<n)-1);
    printf("Case #%d: %d\n", T, ans);
  }
  return 0;
}
