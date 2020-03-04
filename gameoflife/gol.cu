#include<stdio.h>
typedef unsigned cellt;

__global__ void runStep(const cellt *__restrict__ prev, cellt *__restrict__ next, int width, int h, int w) {
  __shared__ cellt buf[34][48];

  int x = threadIdx.x;
  int y = threadIdx.y;

  int col = blockIdx.x * 32 + x;
  int row = blockIdx.y * 32 + y;

  // move to shared memory
  int idx = row * width + col;
  buf[y][x] = prev[idx];
  buf[y][x+16] = prev[idx+16];
  if (x < 2) buf[y][x+32] = prev[idx+32];
  buf[y+16][x] = prev[idx+16*width];
  buf[y+16][x+16] = prev[idx+16*width+16];
  if (x < 2) buf[y+16][x+32] = prev[idx+16*width+32];
  if (y < 2) {
    buf[y+32][x] = prev[idx+32*width];
    buf[y+32][x+16] = prev[idx+32*width+16];
    if (x < 2) buf[y+32][x+32] = prev[idx+32*width+32];
  }
  __syncthreads();

  for (int i = 0; i < 32; i += 16) {
    for (int j = 0; j < 32; j += 16) {
      // compute neighbor sum
      cellt left = buf[y+i][x+0+j] + buf[y+1+i][x+0+j] + buf[y+2+i][x+0+j];
      cellt mid = buf[y+i][x+1+j] + buf[y+1+i][x+1+j] + buf[y+2+i][x+1+j];
      cellt right = buf[y+i][x+2+j] + buf[y+1+i][x+2+j] + buf[y+2+i][x+2+j];
      cellt tot = (left>>28) + (mid<<4) + mid + (mid>>4) + (right<<28);
      cellt me = buf[y+1+i][x+1+j];
      tot -= me;
      // when tot = 0010/0011 and me = 0001 or tot = 0011 and me = 0000
      // => tot|me = 0011
      cellt ans = (tot | me);
      ans = ~(ans>>2) & ans>>1 & ans & 0x11111111;

      // edge case
      if (col+j > (w>>3)) ans = 0;
      if (col+j == (w>>3)) ans &= ~(-1<<(w&7)*4);
      if (row+i < h)
        next[(row+1+i) * width + (col+1+j)] = ans;
    }
  }
}

__global__ void clearEdge(cellt *grid, int width, int w) {
  int y = blockIdx.x * 32 + threadIdx.y * 4 + 1;
  int at = w>>3;
  int x = (at & ~31) + threadIdx.x;
  int idx = y * width + x+1;
  for (int i = 0; i < 4; i++) {
    if (x > at) grid[idx] = 0;
    if (x == at) grid[idx] &= ~(-1<<(w&7)*4);
    idx += width;
  }
}

int h, w, T;
cellt *pin_mem, *gpu_prev, *gpu_next;

int main() {
  cudaDeviceSynchronize();
  fprintf(stderr, "start\n");
  scanf("%d %d %d", &h, &w, &T);
  int ch = getchar();
  while (ch != '\n' && ch != EOF) ch = getchar();

  int blk_x = (w+255) / 256;
  int blk_y = (h+31) / 32;
  int mem_width = blk_x*32+8;
  size_t mem_size = sizeof(cellt) * mem_width * (blk_y*32+2);
  char *str = new char[w+100];
  cudaMallocHost(&pin_mem, mem_size, cudaHostAllocDefault);
  cudaMalloc(&gpu_prev, mem_size);
  cudaMalloc(&gpu_next, mem_size);
  cudaMemset(pin_mem, 0, mem_size);
  cudaDeviceSynchronize();

  for (int i = 0; i < h; i++) {
    fgets(str, w+5, stdin);
    for (int j = 0; j < w; j++) {
      pin_mem[(i+1) * mem_width + (j>>3)+1] |= (str[j]&1)<<(j*4&31);
    }
  }
  cudaMemcpy(gpu_prev, pin_mem, mem_size, cudaMemcpyHostToDevice);
  for (int i = 0; i < T; i++) {
    cellt *g1 = i%2==0 ? gpu_prev : gpu_next;
    cellt *g2 = i%2==0 ? gpu_next : gpu_prev;
    runStep<<<dim3(blk_x, blk_y), dim3(16, 16)>>>(g1, g2, mem_width, h, w);

    // clear edge x > w
    //clearEdge<<<dim3(blk_y), dim3(32, 8)>>>(g2, mem_width, w);
    // clear edge y > h
    //cudaMemset(g2 + (h+1) * mem_width, 0, mem_size - sizeof(cellt) * (h+1) * mem_width);
    //cudaDeviceSynchronize();
  }
  cudaMemcpy(pin_mem, T%2==0 ? gpu_prev : gpu_next, mem_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      int bit = pin_mem[(i+1) * mem_width + (j>>3)+1]>>(j*4&31) & 15;
      str[j] = '0' + bit;
    }
    str[w] = '\0';
    puts(str);
  }
  delete[] str;
  cudaFree(gpu_next);
  cudaFree(gpu_prev);
  cudaFreeHost(pin_mem);
  fprintf(stderr, "finish\n");
}
