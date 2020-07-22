#ifdef _WIN32
#error This program uses UNIX specific functions, and you do not need this program on Windows anyways.
#endif
#include <csignal>
#include <stdio.h>
#include <ctime>

volatile int whichsig;

void handler(int signum) {
  whichsig = signum;
}

int main() {
  int cnt;
  cudaGetDeviceCount(&cnt);
  for(int i = 0; i < cnt; i++){
    cudaSetDevice(i);
  }
  FILE *f = fopen("persist.txt", "w");
  fprintf(f, "start persist.cu\n");
  fflush(f);
  for (int i = 0; i < 64; i++) {
    struct sigaction act, old_act;
    act.sa_handler = handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = 0;
    sigaction(i, &act, &old_act);
  }
  for (;;) {
    struct timespec tm = {1, 0}, rem;
    if (nanosleep(&tm, &rem) == -1) {
      break;
    }
  }
  fprintf(f, "caught signal %d\n", whichsig);
  fclose(f);
  return 0;
}
