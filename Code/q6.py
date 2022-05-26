import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy import stats


def gaussian(data):
    mu = sum(data) / len(data)
    cov = np.cov(data)
    print(mu)
    print(cov)
    xs = np.linspace(0, 20, 40)
    ys = stats.norm.pdf(xs, mu, cov**.5)
    plt.plot(xs, ys, color="r")


def main():
    fp = open("data6.txt", "r")
    data = [float(line.strip()) for line in fp]  
    print(data)
    
    # Plots histogram
    ran = 2*m.ceil(max(data)) - m.floor(min(data))
    plt.hist(data, ran, density=True, color="c")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.title("Rayleigh Distribution")
    
    theta = len(data) / sum([x*x for x in data])
    print(theta)
    xs = np.linspace(0, 20, 40)
    ys = []
    for x in xs:
        y = 2*theta*x*m.e**(-theta*x*x)
        ys.append(y)
    plt.plot(xs, ys, color="k")
    
    gaussian(data)


if __name__ == "__main__":
    main()