import numba
import numpy as np
import time
from numba import cuda, float32, uint32, int32

@cuda.jit(func_or_sig=(numba.types.Array(int32,ndim=1,layout="F"),int32,numba.types.Array(int32,ndim=1,layout="F")))
def move_pp(arr, n, out):
    ptr = cuda.grid(1)
    sum=0
    for i in range(n):
        #sum += ptr
        ptr = arr[ptr]
    out[cuda.grid(1)] = ptr
numba.cuda.profiling()
grid_size = 20
block_size = 128
N = 4000000
ra = np.arange(N)
np.random.shuffle(ra)
rr = np.zeros(N, dtype=np.int)
for i in range(N-1):
    rr[ra[i]] = ra[i+1]
rr[ra[N-1]] = ra[0]
gg = cuda.to_device(rr)
ggo = cuda.device_array(grid_size * block_size, np.int)
move_pp[grid_size, block_size](gg, 0, ggo)
cuda.synchronize()
start = time.time()
move_pp[grid_size, block_size](gg, N, ggo)
cuda.synchronize()
end = time.time()
out = ggo.copy_to_host()
print(out)
speeda = N * 4 * grid_size * block_size / (end - start)
print("%f MB/s" % (speeda / 1000000))
