import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
import math

def chernoff():
    B = [i/100 for i in range(0, 101, 1)]
    y = []
    for b in B:
        y_B = math.e**(-16*b + 16*b**2)
        y.append(y_B)
    
    bs = y.index(min(y))/100
    print("Error:", min(y), "B*:", bs)
    plt.xlabel("B")
    plt.ylabel("Error Bound")
    plt.title("Chernoff Bound")
    plt.plot(B, y)
    s="B*=" + str(bs) + "\nCB=" + str(round(min(y),4))
    plt.plot(bs, min(y), "*k", label=s)
    plt.legend()


def get_points(mean, cov, n):
    vals1 = np.random.multivariate_normal(mean, cov, n)
    x1 = [v[0] for v in vals1]
    x2 = [v[1] for v in vals1]
    return x1, x2

def error_rate(x11, x12, x21, x22, pr=False):
    # Creates the confusion matrix
    co1, inc1, co2, inc2 = 0,0,0,0
    for i in range(len(x11)):
        v = x11[i] + x12[i]
        if v > 4: # It's class1
            co1 += 1
        else:
            inc1 += 1
    for i in range(len(x22)):
        v = x21[i] + x22[i]
        if v <= 4: #Its class2
            co2 += 1
        else:
            inc2 += 1
    if pr:
        print(co1, inc1)
        print(co2, inc2)
    
    # Calculates error rate
    return (inc1 + inc2) / (co1 + inc1 + inc2 + co2)


def main():
    chernoff()
    
    x11, x12 = get_points([4, 4], [[1, 0], [0, 1]], 25)
    x21, x22 = get_points([0, 0], [[1, 0], [0, 1]], 25)

    fig, ax = plt.subplots()
    ax.scatter(x11, x12, label="\u03C91")
    ax.scatter(x21, x22, label="\u03C92")
    
    x = np.linspace(-3, 6, 90)
    db = [4-p for p in x]
    
    ax.plot(x, db, "g", label="DB")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.set_title("Bivariate Gaussian Distribution")
    
    eer = error_rate(x11, x12, x21, x22, True)
    print("\nError:", eer, "(n=25)")
    
    # Checks the empirical error rate
    for j in range(3):
        error_n, n = [], []
        for i in range(100, 1100, 100):
            x11, x12 = get_points([4, 4], [[1, 0], [0, 1]], i)
            x21, x22 = get_points([0, 0], [[1, 0], [0, 1]], i)
            eer = error_rate(x11, x12, x21, x22)
            error_n.append(eer)
            n.append(i)
            print("Error:", eer, "(n=" + str(i) + ")")
        
        # plots the error rate as EER(n)
        fig1, ax1 = plt.subplots()
        ax1.plot(n, error_n)
        ax1.set_xlabel("n")
        ax1.set_ylabel("Error Rate")
        ax1.set_title("Empirical Error Rate")

if __name__ == "__main__":
    main()