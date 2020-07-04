ncr = [[1]]
ss = [0]*33
for i in range(32):
    rr = [1]
    for j in range(i):
        rr.append(ncr[i][j] + ncr[i][j+1])
    rr.append(1)
    ncr.append(rr)

def par(n, mx, arr, size):
    if n == 0:
        prod=1
        cnt=0
        for i in arr:
            cnt += i
            prod *= ncr[cnt][i]
        cnt=0
        tot=size
        prev=arr[0]
        for i in arr:
            if i != prev:
                prod *= ncr[tot][cnt]
                tot -= cnt
                prev = i
                cnt = 0
            cnt += 1
        prod *= ncr[tot][cnt]
        global ss
        ss[arr[0]] += prod
    for i in range(1, min(n,mx)+1):
        arr.append(i)
        par(n-i, i, arr, size)
        arr.pop()

warpsize = 32
par(warpsize, warpsize, [], warpsize)
E = 0
for i in range(1,warpsize+1):
    E += i * ss[i]
    print(i, ss[i])
print(E / (warpsize**warpsize))
