import random
# n = 2~30
G = []
n = 30
for i in range(n):
    G.append([0] * n)
    for j in range(n):
        if j < i:
            r = random.randint(0, 1)
            G[i][j] = G[j][i] = r

f = open("test.txt", "w")
f.write(str(n) + '\n')
for gr in G:
    w = [str(x) for x in gr]
    f.write(" ".join(w) + '\n')
f.close()
