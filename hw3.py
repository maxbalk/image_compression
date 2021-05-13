import numpy as np
from math import sqrt, erf
from numpy.lib.scimath import log2
import matplotlib.pyplot as plt

Xmin = -10
Xmax = 10
mean = 0
sigma = 10

def p(x): # PDF for gaussian dist
    return (1 / (sigma * sqrt(np.pi * 2))) * np.power(np.e, (-0.5 * ((x-mean)/sigma)**2))

def cdf(x): # CDF for gaussian dist
    return 0.5 * (1 + erf((x-mean)/(sigma * sqrt(2))))

def centroid_condition(t0, t1):
    measure = (cdf(t1) - cdf(t0))
    top = p(t1)*(t1**2 / 2) - p(t0)*(t0**2 / 2)
    return top/measure

def distortion(centroids, boundaries):
    total_distortion = 0
    for idx in range(len(boundaries)-1):
        t0 = boundaries[idx]
        t1 = boundaries[idx+1]
        x_hat = centroids[idx]
        dist_on_interval = p(t1)*(((-x_hat + t1)**3)/3) - p(t0)*(((-x_hat+t0)**3)/3)
        total_distortion += dist_on_interval
    return total_distortion

def continuous_entropy(boundaries):
    total_bitrate = 0
    for idx in range(len(boundaries)-1):
        t0 = boundaries[idx]
        t1 = boundaries[idx+1]
        measure = cdf(t1) - cdf(t0)
        total_bitrate += measure * log2(1/measure)
    return total_bitrate
        

M_max = 100
rates = []
distortions = []

#outer loop
for M in range(1,M_max):
    t = [] # set of decision boundaries
    t.append(Xmin)

    if M == 1: #choose arbitrary initial decision boundary
        t.append(np.random.randint(low=Xmin, high=Xmax))
    else:
        for i in range(len(centroids)):
            if i == 0:
                boundary = (Xmin + centroids[i])/2
            elif i == len(centroids)-1:
                boundary = (centroids[i] + Xmax)/2
            else:
                boundary = (centroids[i] + centroids[i+1])/2
            t.append(boundary)

    t.append(Xmax)
    centroids = []
    for j in range(0, len(t)-1): # for each decision region
        x_hat = centroid_condition(t[j], t[j+1])
        centroids.append(x_hat)
    D = distortion(centroids, boundaries=t)
    R = continuous_entropy(boundaries=t)
    rates.append(R)
    distortions.append(D)

plt.plot(rates, distortions)
plt.show()
    