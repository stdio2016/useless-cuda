#include<stdio.h>
void pattern(int *__restrict a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = (long long)i*77%n;
  }
}
void torture(int *__restrict a, int *__restrict b, int *__restrict c, int n) {
  int s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  int i = 0;
  int b1=0,b2=0,b3=0,b4=0;
  for (i = i; i < n-3; i+=4) {
    int aa1 = a[i], aa2 = a[i+1], aa3 = a[i+2], aa4 = a[i+3];
    s1 += b1;
    b1 = b[aa1];
    s2 += b2;
    b2 = b[aa2];
    s3 += b3;
    b3 = b[aa3];
    s4 += b4;
    b4 = b[aa4];
  }
  s1 += b1; s2+=b2; s3+=b3; s4+=b4;
  c[0] = s1 + s2 + s3 + s4;
}
int main(){
  int *a, *b, c, d;
  int n = 100000000;
  a = new int[n];
  b = new int[n];
  pattern(a, n);
  printf("%d\n", a[1]);
  for (int i = 0; i < 10; i++)
    torture(a, b, &c, n);
  printf("%d\n", c);
}