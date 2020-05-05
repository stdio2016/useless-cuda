#include<iostream>
#include<cstdlib>
#include<algorithm>
int n;
unsigned E[32];
int s[32];
int smallCase() {
  for (int i = 0; i < 1<<n; i++) {
    int z = __builtin_popcount(i);
    int sum = 0;
    for (int j = 0; j < n; j++) {
      if (i>>j & 1)
        sum += __builtin_popcount(i & E[j]);
    }
    sum >>= 1;
    if (sum < s[z]) s[z] = sum;
  }
}

__attribute__((target("popcnt")))
int main(){
  std::cin >> n;
  if (n < 0 || n > 32) exit(1);
  srand(n);
  for(int i=0;i<n;i++)for(int j=0;j<n;j++){
    if(i>j) {
      int b = rand()>>7&1;
      if (b) b = 1;
      E[i]|=b<<j;
      E[j]|=b<<i;
    }
  }
  for(int i=0;i<n;i++){
    int good = 0, y = 0;
    for(int j=0;j<n;j++){
      int w;
      if (std::cin>>w) good = 1;
      y|=w<<j;
    }
    if (good) E[i] = y;
  }
  for(int i=0;i<=n;i++)s[i] = 100000;
  int good = 0;
  int bu[16] = {0};
  for (int i=1;i<16;i++){
    bu[i] = __builtin_popcount(E[__builtin_ctz(i)] & (i ^ i>>1));
  }
  if (n < 4) {
    smallCase();
  }
  else for(int i=0;i<1<<(n-4);i++){
    unsigned a = (i ^ i>>1);
    if (i) {
      int z = __builtin_ctz(i);
      int diff = __builtin_popcount(E[z+4] & a<<4);
      if (a>>z&1) good += diff;
      else good -= diff;
    }
    int g = good;
    
    int s0, s1, s2, s3, s4;
    int E0 = __builtin_popcount(E[0] & a<<4);
    int E1 = __builtin_popcount(E[1] & a<<4);
    int E2 = __builtin_popcount(E[2] & a<<4);
    int E3 = __builtin_popcount(E[3] & a<<4);
    int b = __builtin_popcount(a);
    s0 = g;
    g += E0 + bu[1];
    s1 = g;
    g += E1 + bu[2];
    s2 = g;
    g -= E0 + bu[3];
    if (g < s1) s1 = g;

    g += E2 + bu[4];
    if (g < s2) s2 = g;
    g += E0 + bu[5];
    s3 = g;
    g -= E1 + bu[6];
    if (g < s2) s2 = g;
    g -= E0 + bu[7];
    if (g < s1) s1 = g;

    g += E3 + bu[8];
    if (g < s2) s2 = g;
    g += E0 + bu[9];
    if (g < s3) s3 = g;
    g += E1 + bu[10];
    s4 = g;
    g -= E0 + bu[11];
    if (g < s3) s3 = g;

    g -= E2 + bu[12];
    if (g < s2) s2 = g;
    g += E0 + bu[13];
    if (g < s3) s3 = g;
    g -= E1 + bu[14];
    if (g < s2) s2 = g;
    g -= E0 + bu[15];
    if (g < s1) s1 = g;

    if (s0 < s[b]) s[b] = s0;
    if (s1 < s[b+1]) s[b+1] = s1;
    if (s2 < s[b+2]) s[b+2] = s2;
    if (s3 < s[b+3]) s[b+3] = s3;
    if (s4 < s[b+4]) s[b+4] = s4;
  }
  for (int i = 2; i <= n ; i++) {
    std::cout << i << ' ' << s[i] << '\n';
  }
}