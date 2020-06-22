# 1 test 14ms
# 321ms
# 289ms
# 169ms
# 172ms
# n = 4, 6: use function, only 2 solutions
# n = 5, 7, 8: use lookup, because parallel is slower
# n = 9~11: parallel is only a little faster (< 2x), speed: -p > -f -p > -f > normal
# n = 12: parallel is 4x faster (-p -f 5x faster)
n = 10
s = str(n) + '\n'
for i in range(n):
    row = ['.'] * n
    #if i<5:
    #    row = ['*']*n
    #    row[i*2+1] = '.'
    s = s + ''.join(row) + '\n'
f = open("10.txt", "w")
for i in range(6000):
    f.write(s)
