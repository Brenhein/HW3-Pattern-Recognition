import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy import stats
from sympy import *


def mle_mean(data):
    s = np.sum(data, axis=0)
    s = [s[0] / len(data), s[1] / len(data)]
    return s


def mle_cov(data):
    mu = mle_mean(data)
    pro = []
    for x in data:
        diff = np.subtract(x,mu)  # x-u
        p = np.outer(diff, diff)  # (x-u)(x-u)^t
        pro.append(p)
    sig = sum(pro)  # sum (x-u)(x-u)^t
    return sig/len(pro)
    

def main():
    mu1, cov1 = [0,0], [[1, 0], [0, 1]]
    mu2, cov2 = [5,5], [[1, 0], [0, 1]]
    
    # opens the files to get the data
    ns = ["50", "500", "50000"]
    for n in ns:
        fp1 = open("gen1_" + n + ".txt")
        fp2 = open("gen2_" + n + ".txt")
        w1, w2 = [], []
        for line in fp1:
            line = [float(x) for x in line.strip().split(",")]
            w1.append(line)
        for line in fp2:
            line = [float(x) for x in line.strip().split(",")]
            w2.append(line)
        w1, w2 = np.array(w1), np.array(w2)
    
       
        mu = mle_mean(w1)
        cov = mle_cov(w1)
        print("Mean:", np.array([round(mu[0], 4), round(mu[1], 4)]))
        print("Covariance:\n", cov)
        
        mu = mle_mean(w2)
        cov = mle_cov(w2)
        print("Mean:",  np.array([round(mu[0], 4), round(mu[1], 4)]))
        print("Covariance:\n", cov)
        
        # Gets the class points
        w11 = [x[0] for x in w1]
        w12 = [x[1] for x in w1]
        w21 = [x[0] for x in w2]
        w22 = [x[1] for x in w2]
        xt = np.linspace(-3, 7, 100)
        yt = [5-x for x in xt]
        yb = []
        for x in xt:
            x2 = symbols('x2')
            if n == "50":
                y = solve(.2152 -.1223*x**2 + .2495*x*x2 - 10.7537*x + \
                    48.104 - .127*x2**2 - 7.81769*x2)
                yb.append(y[1])
            elif n == "500":
                y = solve(50.6018 + 0.0217*x**2 - 0.0809*x*x2 - 10.057*x - \
                    0.0737*x2**2 - 10.110*x2)
                yb.append(y[1])
            elif n == "50000":
                y = solve(50.18766 + 0.00417*x**2 + 0.00197*x*x2 - 10.071*x \
                    -0.00301*x2**2 - 10.011*x2)
                yb.append(y[1])
            
        
        # Plots the estimated boundary
        fig, ax = plt.subplots()
        ax.scatter(w11, w12, color="r")
        ax.scatter(w21, w22, color="b")
        ax.plot(xt, yb, color="g", label="Estimated DB")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Bivariate Gaussian Distribution")
        ax.plot(xt, yt, color="k", label="True DB")
        ax.legend()
        
        # Finds the error rate
        err_t, err_b = 0, 0
        for x in w1:
            x, x2 = x[0], x[1]
            if n == "50":
                if ( -.1223*x**2 + .2495*x*x2 - 10.7537*x - \
                     .127*x2**2 - 7.81769*x2) <= -48.104-.2152:
                        err_b +=1    
            elif n == "500":
                if (0.0217*x**2 - 0.0809*x*x2 - 10.057*x - \
                    0.0737*x2**2 - 10.110*x2) <= -50.6018:
                        err_b += 1
            elif n == "50000":
                if (0.00417*x**2 + 0.00197*x*x2 - 10.071*x - \
                    0.00301*x2**2 - 10.011*x2) <= -50.1876:
                        err_b += 1            
            if x + x2 >= 5:
                err_t += 1
                
        for x in w2:
            x, x2 = x[0], x[1]
            if n == "50":
                if ( -.1223*x**2 + .2495*x*x2 - 10.7537*x - \
                     .127*x2**2 - 7.81769*x2) > -48.104-.2152:
                        err_b +=1    
            elif n == "500":
                if (0.0217*x**2 - 0.0809*x*x2 - 10.057*x - \
                    0.0737*x2**2 - 10.110*x2) > -50.6018:
                        err_b += 1
            elif n == "50000":
                if (0.00417*x**2 + 0.00197*x*x2 -10.071*x - \
                    0.00301*x2**2 - 10.011*x2) > -50.1876:
                        err_b += 1
            if x + x2 < 5:
                err_t += 1
                    
        print("Error True:", err_t / (2*int(n)))
        print("Error Estimate:", err_b / (2*int(n)), "\n")
        
    

if __name__ == "__main__":
    main()