#include<stdio.h>
void pattern(int *__restrict a, int n, int m) {
  for (int i = 0; i < n; i++) {
    a[i] = (long long)i*77%m;
  }
}
void torture(int *__restrict a, int *__restrict b, int *__restrict c, int n) {
  int i = 0;
  int s = 0;
  for (i = i; i < n; i++) {
    s += b[a[i]];
  }
  c[0] = s;
}
int main(){
  int *a, *b, c, d;
  int n = 100000, m = 100000;
  a = new int[n];
  b = new int[m];
  pattern(a, n, m);
  printf("%d\n", a[1]);
  for (int i = 0; i < 3000; i++)
    torture(a, b, &c, n);
  printf("%d\n", c);
}