import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp


def main():
    mu1, cov1 = [-1, -1], [[1, 0], [0, 1]]
    mu2, cov2 = [1, 1], [[1, 0], [0, 1]]
    mu3, cov3 = [.5, .5], [[1, 0], [0, 1]]
    mu4, cov4 = [-.5, -.5], [[1, 0], [0, 1]]
    
    fig, ax = plt.subplots()
    ax.scatter(mu1[0], mu1[1], label="\u03C91")
    ax.scatter(mu2[0], mu2[1], label="\u03C92")
    ax.scatter(mu3[0], mu3[1], label="\u03C93,1")
    ax.scatter(mu4[0], mu4[1], label="\u03C93,2")
    ax.scatter(.1, .1, label="x: [.1, .1]")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Bivariate Gaussian Distribution")
    ax.legend()
    
    # Assigns a point
    prob = []
    v = [.1, .1]
    prob.append(sp.multivariate_normal.pdf(v, mu1, cov1))
    prob.append(sp.multivariate_normal.pdf(v, mu2, cov2))
    prob.append(.5*sp.multivariate_normal.pdf(v, mu3, cov3) + \
                .5*sp.multivariate_normal.pdf(v, mu4, cov4))
    print("Densities:", prob)
    print("Max:", max(prob), "(\u03C9" + str(prob.index(max(prob))+1) +")")

if __name__ =="__main__":
    main()
