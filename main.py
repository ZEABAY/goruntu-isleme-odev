# Zeynel Abiddin Aydar 02205076039


import cv2
import numpy as np
import matplotlib.pyplot as plt



img1 = cv2.imread("img1.jpg")
shape = img1.shape
cv2.imshow('img1', img1)


resim_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow('resim gri', resim_gray)


H = np.zeros(shape=(256,1))
for i in range(shape[0]):
    for j in range(shape[1]):
        k = resim_gray[i,j]
        H[k,0] = H[k,0]+1


plt.plot(H)
plt.show()
cv2.waitKey(0)


########################################################################################################################



img2 = cv2.imread('img2.jpg')
cv2.imshow('resim', img2)


img_0 = cv2.imread('img2.jpg',0)
cv2.imshow('resim-0', img_0)


[h, w] = img_0.shape
img2 = np.zeros([h, w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        img2[i, j] = 255 - img_0[i, j]



cv2.imshow("Ters resim", img2)
cv2.waitKey()

