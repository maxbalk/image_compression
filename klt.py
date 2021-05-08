import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread('lena.jpg',0)
img = img.astype(np.double)

"""
KLT REGION
"""
blocks = np.reshape(img, (16, 16384))
Cx = np.cov(blocks)
u,s,v = np.linalg.svd(Cx)
# v is the KLT transform matrix
basis = np.mean(v, axis=1)
#plot basis against on range of 16, value is concentrated toward the smallest index.
plt.plot(range(16), basis)
plt.show()
"""
END KLT REGION
"""

"""FILTERING REGION"""
