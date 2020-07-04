import numpy as np
from numba import cuda

@cuda.jit
def test_block_size(success, good):
    success[0] = good
    a = cuda.ballot_sync(9487, cuda.blockIdx.x == 10)
    a = cuda.ballot_sync(9487, a)
    a = cuda.ballot_sync(9487, a)
    a = cuda.ballot_sync(9487, a)
    a = cuda.ballot_sync(9487, a)
    a = cuda.ballot_sync(9487, a)
    a = cuda.ballot_sync(9487, a)

arr = np.arange(32)
arr[0] = 9487
d_arr = cuda.to_device(arr)

good = 0
for x in range(1,1026):
    print("x=%d" % x)
    for y in range(1,1026):
        can_xy = False
        for z in range(1,1026):
            good += 1
            try:
                test_block_size[1, (x, y, z)](d_arr, good)
                d_arr.copy_to_host(arr)
                if not arr[0] == good:
                    good -= 1
                    break
                else:
                    can_xy = True
            except cuda.cudadrv.driver.CudaAPIError:
                good -= 1
                break
        if not can_xy:
            break
print(good)
