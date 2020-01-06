#include<stdio.h>
typedef signed i32;
typedef unsigned u32;
typedef unsigned long long u64;
int findlight(int n, int m) {
    u64 hi = u64(1)<<n-1;
    u64 mask = (u64(2)<<n-1)-1;
    int ans = 0;
    for (u64 row = 0; row < hi; row++) {
        u64 up = 0;
        u64 tmp = row<<1;
        for (int i = 0; i < m; i++) {
            u64 left = tmp<<1;
            u64 right = tmp>>1;
            u64 down = ~(tmp ^ left ^ right ^ up) & mask;
            up = tmp;
            tmp = down;
        }
        if (tmp == 0) ans += 1;
        up = 0;
        tmp = row<<1|1;
        for (int i = 0; i < m; i++) {
            u64 left = tmp<<1;
            u64 right = tmp>>1;
            u64 down = ~(tmp ^ left ^ right ^ up) & mask;
            up = tmp;
            tmp = down;
        }
        if (tmp == 0) ans += 1;
    }
    return ans;
}
int main() {
    int n, m;
    printf("light out solution count\n");
    printf("enter n(1~64), m(1+): ");
    scanf("%d %d", &n, &m);
    if (n < 1 || n > 64) return 1;
    if (m < 1) return 1;
    printf("solution count = %d\n", findlight(n, m));
}