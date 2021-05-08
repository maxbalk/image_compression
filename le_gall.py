import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread('lena.jpg',0)
img = img.astype(np.double)


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
    bands = np.rot90(bands, -1)
    bands = decompose(bands)
    bands = np.rot90(bands, 1)
    return bands

decomp_level = 1
y_out = img
for i in range(decomp_level):
    x_dim = y_out.shape[0]
    y_dim = y_out.shape[1]
    y_out = analysis(y_out)
    #y_out = y_out[:int(x_dim/2),:int(y_dim/2)]

plt.imshow(y_out)
plt.show()

def reconstruct(img):
    rows = []
    for row in img:
        # upsample. do first and second halves serparately 
        g0 = np.insert(row, range(1,len(row)+1), 0)
        g1 = np.insert(row, range(1,len(row)+1), 0)
        # apply filter
        g0 = transform(g0, hi_inverse)
        g1 = transform(g1, lo_inverse)
        # add row values together, not append
        rows.append(np.append(g0,g1))
    return np.array(rows)

def synthesis(img):
    bands = reconstruct(img)
    bands = np.rot90(bands, -1)
    bands = reconstruct(bands)
    bands = np.rot90(bands, 1)
    return bands

x_hat = y_out
for i in range(decomp_level):
    x_hat = synthesis(x_hat)
    x_dim = x_hat.shape[0]
    y_dim = x_hat.shape[1]
    #x_hat = x_hat[:int(x_dim/2),:int(y_dim/2)]

plt.imshow(x_hat)
plt.show()

mse = (np.square(img - x_hat)).mean(axis=None)
print(mse)
