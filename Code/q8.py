import numpy as np
import scipy.stats as sp
import pandas as pd


def confusion(predicted, actual):
    actual = pd.Series(actual, name="actual")
    predicted = pd.Series(predicted, name="predicted")
    confuse = pd.crosstab(actual, predicted)
    print(confuse)
    
    # finds error rate
    err = (confuse[1][2] + confuse[1][3] + confuse[2][1] + confuse[2][3] +\
           confuse[3][1] + confuse[3][2]) / len(actual)
    print("Error Rate:", round(err, 4))


def classify(pattern, mus, covs):
    max_p, max_i = -1, -1
    pattern = np.array(pattern)
    
    for i in range(len(covs)):
        # Calculates the condiitonal probability (equal priors and 0-1 loss)
        mu, cov = mus[i], covs[i]
        density = sp.multivariate_normal.pdf(pattern, mu, cov)
        if density > max_p:
            max_p = density
            max_i = i
    print("Class:", max_i+1)
    return max_i + 1
    

def find_mean(iris):
    pat = [iris[1][:25], iris[2][:25], iris[3][:25]]
    means = []
    for i, p in enumerate(pat):
        m = np.mean(p, axis=0)
        means.append(m)
        print("Means:", m)
    return means[0], means[1], means[2]
    

def find_var(iris):
    pat = [iris[1][:25], iris[2][:25], iris[3][:25]]
    for i in range(len(pat)):
        pat[i] = np.array(pat[i]).T
        
    covs = []
    for i, p in enumerate(pat):
        cov = np.cov(p) 
        covs.append(cov)
        print("Covariance:\n", cov, "\n")
    return covs[0], covs[1], covs[2]


def main():
    fp = open("iris.txt", "r")
    lines = []
    for line in fp:
        line = line.strip().split()
        for i in range(len(line)):
            if i == 4:
                line[i] = int(line[i])
            else:
                line[i] = float(line[i])
        if len(line) == 5:
            lines.append(line)
    line = line[:-1]
    
    # Turns into a dict
    iris = {}
    for el in lines:
        if el[4] not in iris:
            iris[el[4]] = [el[:-1]]
        else:
            iris[el[4]].append(el[:-1])
            
    mus = find_mean(iris)
    covs = find_var(iris)
    
    # Classifies the remaining patterns
    pat_test = [iris[1][25:], iris[2][25:], iris[3][25:]]
    actual = [1] * 25 + [2] * 25 + [3] * 25
    predicted = []
    for i, patts in enumerate(pat_test):
        for p in patts:
            res = classify(p, mus, covs)
            predicted.append(res)
    confusion(predicted, actual)
            
    
if __name__ == "__main__":
    main()