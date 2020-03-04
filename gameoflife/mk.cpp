#include <stdio.h>
#include <cstdlib>
char buf[3000];
int main(int argc, char *argv[]) {
  int n = 10, n2 = 10, m = 1;
  if (argc > 1) sscanf(argv[1], "%d", &n);
  if (argc > 2) sscanf(argv[2], "%d", &n2);
  if (argc > 3) sscanf(argv[3], "%d", &m);
  int seed = 1;
  if (argc > 4) sscanf(argv[4], "%d", &seed);
  srand(seed);
  printf("%d %d %d\n",n,n2,m);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n2; j++) buf[j] = (rand()>>5&1) + '0';
    puts(buf);
  }
}