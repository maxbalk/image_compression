import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread('lena.jpg',0)
img = img.astype(np.double)
from copy import deepcopy, copy

# define filter coefficients
lo_forward = np.array([-1/8, 2/8, 6/8, 2/8, -1/8])
hi_forward = np.array([-1/2, 1, -1/2])
hi_inverse = np.array([1/2, 1, 1/2])
lo_inverse = np.array([-1/8, -2/8, 6/8, -2/8, -1/8])

def transform(x, filter):
    extension = math.floor(len(filter)/2)
    x = np.insert(x, 0, x[1:1+extension][::-1])
    x = np.append(x, x[len(x)-(extension)-1:-1][::-1])
    out = []
    for i in range(extension, len(x)-extension):
        out.append(np.dot(filter, x[i-extension:i+extension+1]))
    return np.array(out)

def decompose(img):
    rows = []
    for row in img:
        # apply filter
        h0 = transform(row, lo_forward)
        h1 = transform(row, hi_forward)
        # downsample 
        h0 = h0[0::2]
        h1 = h1[0::2]
        rows.append(np.append(h0, h1))
    return np.array(rows)

def analysis(img):
    bands = decompose(img)
    bands = np.rot90(bands, 1)
    bands = decompose(bands)
    bands = np.rot90(bands, -1)
    return bands

decomp_level = 3
y_out = deepcopy(img)
x_dim = y_out.shape[0]
y_dim = y_out.shape[1]
for i in range(decomp_level):    
    y_out[:int(x_dim/(2**i)),:int(y_dim/(2**i))] = analysis(y_out[:int(x_dim/(2**i)),:int(y_dim/(2**i))])

# plt.imshow(y_out)
# plt.show()

def reconstruct(img):
    rows = []
    for i, row in enumerate(img):
        g0 = row[:int(len(row)/2)]
        g1 = row[int(len(row)/2):]
        # upsample. do first and second halves serparately 
        g0 = np.insert(g0, range(1,len(g0)+1), 0)
        g1 = np.insert(g1, range(1,len(g1)+1), 0)
        # apply filter
        g0 = transform(g0, hi_inverse)
        g1 = transform(g1, lo_inverse)
        # add row values together, not append
        img[i] = np.sum([g0,g1], axis=0)
    return img

def synthesis(img):
    bands = np.rot90(img, 1)
    bands = reconstruct(bands)
    bands = np.rot90(bands, -1)
    bands = reconstruct(bands)
    return bands

x_hat = deepcopy(y_out)
for i in reversed(range(decomp_level)):
    x_hat[:int(x_dim/(2**i)),:int(y_dim/(2**i))] = synthesis(x_hat[:int(x_dim/(2**i)),:int(y_dim/(2**i))])

plt.imshow(x_hat)
plt.show()

mse = (np.square(img - x_hat)).mean(axis=None)
print(mse)
