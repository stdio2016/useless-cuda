// pentomino puzzle gpu acceleration proof of concept
// use bfs
// by stdio2016 2023-03-22
#include <cstdio>
#include <set>
#include <algorithm>
#include <vector>
#include <cstdint>
#ifdef _WIN32
#include <intrin.h>
#endif

typedef std::pair<int,int> coord;

char name[12] = {
  'I', 'L', 'N', 'Y', 'P', 'U',
  'V', 'W', 'Z', 'T', 'F', 'X'
};

int ctzll(uint64_t num) {
#if __GNUC__
  return __builtin_ctzll(num);
#elif _WIN64
  unsigned long ans = 0;
  if (_BitScanForward64(&ans, num)) {
    return ans;
  }
  return 64;
#else
  unsigned long ans = 0;
  for (ans = 0; ans < 64; ans++) {
    if (num>>ans & 1) {
      return ans;
    }
  }
  return ans;
#endif
}

struct SS {
  std::pair<int, int> p[5];
  void rotate() {
    int maxx = 0;
    for (int i = 0; i < 5; i++)
      maxx = std::max(maxx, p[i].first);
    for (int i = 0; i < 5; i++) {
      p[i] = coord{p[i].second, maxx-p[i].first};
    }
    std::sort(p, p+5);
  }
  
  void mirror() {
    int maxx = 0;
    for (int i = 0; i < 5; i++)
      maxx = std::max(maxx, p[i].first);
    for (int i = 0; i < 5; i++) {
      p[i] = coord{maxx-p[i].first, p[i].second};
    }
    std::sort(p, p+5);
  }
};
bool operator<(SS a,SS b) {
  for (int i = 0; i < 5; i++) {
    if (a.p[i] < b.p[i]) return true;
    if (a.p[i] > b.p[i]) return false;
  }
  return false;
}

coord shape[12][5] = {
  {{0,0},{1,0},{2,0},{3,0},{4,0}},
  {{0,0},{1,0},{2,0},{3,0},{3,1}},
  {{0,0},{1,0},{2,0},{2,1},{3,1}},
  {{0,0},{1,0},{2,0},{2,1},{3,0}},
  {{0,0},{1,0},{1,1},{2,0},{2,1}},
  {{0,0},{0,1},{1,0},{2,0},{2,1}},
  
  {{0,0},{1,0},{2,0},{2,1},{2,2}},
  {{0,0},{1,0},{1,1},{2,1},{2,2}},
  {{0,0},{1,0},{1,1},{1,2},{2,2}},
  {{0,0},{1,0},{1,1},{1,2},{2,0}},
  {{0,1},{1,0},{1,1},{1,2},{2,0}},
  {{0,1},{1,0},{1,1},{1,2},{2,1}}
};

std::vector<uint64_t> imagelist[64][12];

void solve(uint64_t space, unsigned unused, int &ans) {
  if (unused == 0) {
    ans += 1;
    return;
  }
  ans ++;
  int choose = ctzll(space);
  for (int i = 0; i < 12; i++) {
    if (unused >> i & 1) {
      for (uint64_t m : imagelist[choose][i]) {
        if ((space & m) == m) {
          solve(space - m, unused - (1U<<i), ans);
        }
      }
    }
  }
}

void solve_bfs(uint64_t space, unsigned unused, int &ans) {
  std::vector<uint64_t> cur_space(1, space);
  std::vector<unsigned> cur_unused(1, unused);
  std::vector<uint64_t> nxt_space;
  std::vector<unsigned> nxt_unused;
  while (!cur_space.empty()) {
    for (size_t i = 0; i < cur_space.size(); i++) {
      space = cur_space[i];
      unused = cur_unused[i];
      ans++;
      int choose = ctzll(space);
      for (int j = 0; j < 12; j++) {
        if (unused >> j & 1) {
          for (uint64_t m : imagelist[choose][j]) {
            if ((space & m) == m) {
              nxt_space.push_back(space - m);
              nxt_unused.push_back(unused - (1U<<j));
            }
          }
        }
      }
    }
    std::swap(cur_space, nxt_space);
    nxt_space.clear();
    std::swap(cur_unused, nxt_unused);
    nxt_unused.clear();
  }
}

__global__ void solve_gpu(
  uint64_t *imagelist, int *imgpos,
  uint64_t *cur_space, unsigned *cur_unused,
  uint64_t *nxt_space, unsigned *nxt_unused,
  int n_cur, int *n_next
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cur) {
    return;
  }
  uint64_t space = cur_space[idx];
  unsigned unused = cur_unused[idx];
  int choose = __ffsll(space) - 1;
  for (int i = 0; i < 12; i++) {
    if (unused >> i & 1) {
      int end = imgpos[choose * 12 + i + 1];
      for (int j = imgpos[choose * 12 + i]; j < end; j++) {
        uint64_t m = imagelist[j];
        if ((space & m) == m) {
          int to_insert = atomicAdd(n_next, 1);
          nxt_space[to_insert] = space - m;
          nxt_unused[to_insert] = unused - (1U<<i);
        }
      }
    }
  }
}

int main() {
  int height = 10, width = 6;
  
  for (int i = 0; i < 12; i++) {
    std::set<SS> rots;
    SS cur;
    for (int j = 0; j < 5; j++) cur.p[j] = shape[i][j];
    for (int r = 0; r < 4; r++) {
      rots.insert(cur);
      cur.rotate();
    }
    cur.mirror();
    for (int r = 0; r < 4; r++) {
      rots.insert(cur);
      cur.rotate();
    }
    
    int id = 1;
    for (SS r: rots) {
      int mx = 0, my = 0;
      for (int j = 0; j < 5; j++) {
        my = std::max(my, r.p[j].first);
        mx = std::max(mx, r.p[j].second);
      }
      for (int y = 0; y < height - my; y++) {
        for (int x = 0; x < width - mx; x++) {
          uint64_t bset = 0;
          for (int j = 0; j < 5; j++) {
            int yy = y + r.p[j].first, xx = x + r.p[j].second;
            bset += 1ULL<<(yy *width + xx);
          }
          //printf("%d %d %llx\n", i, __builtin_ctzll(bset), bset);
          imagelist[ctzll(bset)][i].push_back(bset);
        }
      }
      id++;
    }
  }
  std::vector<uint64_t> flatimglist;
  std::vector<int> imgpos(1, 0);
  for (int i = 0; i < 60; i++) {
    for (int j = 0; j < 12; j++) {
      for (int k = 0; k < imagelist[i][j].size(); k++) {
        flatimglist.push_back(imagelist[i][j][k]);
      }
      imgpos.push_back(flatimglist.size());
    }
  }
  printf("%d\n", (int)flatimglist.size());
  printf("%d\n", (int)imgpos.size());
  uint64_t *gpu_imglist;
  int *gpu_imgpos;
  uint64_t *gpu_space1, *gpu_space2;
  unsigned *gpu_unused1, *gpu_unused2;
  int *gpu_next;
  cudaEvent_t event1, event2;
  cudaEventCreate(&event1);
  cudaEventCreate(&event2);
  cudaEventRecord(event1, 0);
  cudaMalloc(&gpu_imglist, sizeof(uint64_t) * flatimglist.size());
  cudaMalloc(&gpu_imgpos, sizeof(int) * imgpos.size());
  cudaMalloc(&gpu_space1, sizeof(uint64_t) * 2560000);
  cudaMalloc(&gpu_space2, sizeof(uint64_t) * 2560000);
  cudaMalloc(&gpu_unused1, sizeof(int) * 2560000);
  cudaMalloc(&gpu_unused2, sizeof(int) * 2560000);
  cudaMalloc(&gpu_next, sizeof(int));
  cudaMemcpy(gpu_imglist, &flatimglist[0], sizeof(uint64_t) * flatimglist.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_imgpos, &imgpos[0], sizeof(int) * imgpos.size(), cudaMemcpyHostToDevice);
  int ans=0;
  std::vector<uint64_t> cpu_space;
  std::vector<unsigned> cpu_unused(8, 0x7ff);
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 2; x++) {
      unsigned long long bset = (1ULL<<60)-1;
      bset -= (1ULL<<1|1<<6|1<<7|1<<8|1<<13)<<(y * width + x);
      //solve_bfs(bset, 0x7ff, ans);
      cpu_space.push_back(bset);
    }
  }
  ans = cpu_space.size();
  cudaMemcpy(gpu_space1, &cpu_space[0], sizeof(uint64_t) * cpu_space.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_unused1, &cpu_unused[0], sizeof(unsigned) * cpu_unused.size(), cudaMemcpyHostToDevice);
  cudaMemset(gpu_next, 0, sizeof(int));

  for (int step = 2; step <= 12; step++) {
    solve_gpu<<<10000,256>>>(gpu_imglist, gpu_imgpos,
      gpu_space1, gpu_unused1, gpu_space2, gpu_unused2, ans, gpu_next);
    cudaMemcpy(&ans, gpu_next, sizeof(int), cudaMemcpyDeviceToHost);
    printf("step=%d n_next=%d\n", step, ans);
    std::swap(gpu_space1, gpu_space2);
    std::swap(gpu_unused1, gpu_unused2);
    cudaMemset(gpu_next, 0, sizeof(int));
  }

  cudaEventRecord(event2, 0);
  cudaEventSynchronize(event2);
  printf("ans=%d\n", ans);
  float tim;
  cudaEventElapsedTime(&tim, event1, event2);
  printf("time %fms\n", tim);
}
