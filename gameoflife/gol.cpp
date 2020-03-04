#include<stdio.h>
#include<cstring>

int h, w, T;
char *prev, *next;

int main() {
  fprintf(stderr, "start\n");
  scanf("%d %d %d", &h, &w, &T);
  int ch = getchar();
  while (ch != '\n' && ch != EOF) ch = getchar();

  char *str = new char[w+100];
  prev = new char[(h+2)*(w+2)];
  next = new char[(h+2)*(w+2)];
  memset(prev, 0, sizeof(char)*(h+2)*(w+2));
  memset(next, 0, sizeof(char)*(h+2)*(w+2));

  for (int i = 1; i <= h; i++) {
    fgets(str, w+5, stdin);
    for (int j = 1; j <= w; j++) {
      prev[i*(w+2)+j] = str[j-1]&1;
    }
  }
  for (int i = 0; i < T; i++) {
    char *c1 = i%2 == 0 ? prev : next;
    char *c2 = i%2 == 0 ? next : prev;
    #pragma omp parallel for schedule(static)
    for (int y = 1; y <= h; y++) {
      for (int x = 1, idx = y*(w+2)+x; x <= w; x++, idx++) {
        int sum = c1[idx-(w+2)-1] + c1[idx-(w+2)] + c1[idx-(w+2)+1]
          + c1[idx-1] + c1[idx+1]
          + c1[idx+(w+2)-1] + c1[idx+(w+2)] + c1[idx+(w+2)+1];
        if (c1[idx]) c2[idx] = sum == 2 || sum == 3;
        else c2[idx] = sum == 3;
      }
    }
  }
  char *res = T%2 == 0 ? prev : next;
  for (int i = 1; i <= h; i++) {
    for (int j = 1; j <= w; j++) {
      str[j-1] = '0' + res[i*(w+2)+j];
    }
    str[w] = '\0';
    puts(str);
  }
  delete[] str;
  delete[] prev;
  delete[] next;
  fprintf(stderr, "finish\n");
}
