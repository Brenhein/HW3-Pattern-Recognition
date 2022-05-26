import numpy as np
n = int(input("How many samples:"))
mu1, cov1 = [0,0], [[1, 0], [0, 1]]
mu2, cov2 = [5,5], [[1, 0], [0, 1]]
w1 = np.random.multivariate_normal(mu1, cov1, n)
w2 = np.random.multivariate_normal(mu2, cov2, n)
fp1 = open("gen1_" + str(n) + ".txt", "w")
fp2 = open("gen2_" + str(n) + ".txt", "w")
for x1, x2 in w1:
    fp1.write(str(x1)+","+str(x2)+"\n")
for x1, x2 in w2:
    fp2.write(str(x1)+","+str(x2)+"\n")
fp1.close()
fp2.close()