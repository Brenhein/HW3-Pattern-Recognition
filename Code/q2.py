import numpy as np
import  matplotlib.pyplot as plt


def rand_multinorm(mean, cov, aw):
    """Plots the normal bivariate distrubution and the whitened form"""
    vals = np.random.multivariate_normal(mean, cov, 10000)
    x1 = [v[0] for v in vals]
    x2 = [v[1] for v in vals]
    
    
    fig, ax = plt.subplots()
    ax.scatter(x1, x2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Bivariate Gaussian Distribution")
    
    # Gets the whitened points
    w1, w2 = [], []
    for i in range(len(x1)):
        white = np.dot(aw, [[x1[i]], [x2[i]]])
        w1.append(white[0])
        w2.append(white[1])
    
    fig, ax1 = plt.subplots()
    ax1.scatter(w1, w2)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("Whitened Bivariate Gaussian Distribution")


def whiten(mean, cov, mean_t):
    vals, vec = np.linalg.eig(cov)
    
    # Creates matrix of values
    lam = []
    for i in range(len(vals)):
        row = [0] * len(vals)
        row[i] = vals[i]
        lam.append(row)

    lam = np.linalg.inv(lam)
    lam = np.sqrt(lam)
    white = np.dot(vec, lam)
    white_t = np.transpose(white)
    
    print("Whitening:\n", white)
    print()
    print("Whitening Transposed:\n", white_t)
    print()
    mean_w = np.dot(white_t, mean)
    print("Whitened Mean:\n", mean_w)
    print()
    dot = np.dot(white_t, cov)
    dot = np.dot(dot, white)
    print("Whitened Covariance:\n", dot)
    
    rand_multinorm(mean_t, cov, white_t)
  

def main():
    mean = [[0], [0]]
    mean_t = [0, 0]
    cov = [[20, 10], [10, 30]]
    
    whiten(mean, cov, mean_t)
    

if __name__ == "__main__":
    main()