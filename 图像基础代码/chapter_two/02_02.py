#----02 图像滤波与增强-----
# 线性滤波： 方框滤波、均值滤波、高斯滤波
# 非线性滤波： 中值滤波、双边滤波
# 直方图均衡化 gamma变换
# author：RingoWu
# time：2020.7.21

import cv2
import numpy as np
import matplotlib.pyplot as plt

#-----------------------方框滤波  ksize越大越模糊----------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', cv2.IMREAD_UNCHANGED)
    dst = cv2.boxFilter(img,-1,(3,3),normalize=True)
    dst1 = cv2.boxFilter(img,-1,(3,3),normalize=False)
    cv2.imshow('orgin', img)
    cv2.imshow('boxfilter_normalize', dst)
    cv2.imshow('boxfilter', dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-------------------------均值滤波-------------------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #opencv 与 matplotlib显示通道顺序不同
    dst = cv2.blur(img, (3, 3))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("orgin")
    plt.subplot(1,2,2)
    plt.imshow(dst)
    plt.title('mean filter')
    plt.show()

#--------------------------高斯滤波-------------------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', 1)
    dst = cv2.GaussianBlur(img,(3,3),3)
    cv2.imshow('orgin', img)
    cv2.imshow('gaussion filter', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#------------------------中值滤波----------------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('median.png', cv2.IMREAD_UNCHANGED)
    dst = cv2.medianBlur(img,7)
    cv2.imshow('orgin',img)
    cv2.imshow('median filter', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-----------------------双边滤波-----------------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('bilateral.png', 1)
    dst = cv2.bilateralFilter(img,2,9,15)
    cv2.imshow('orgin',img)
    cv2.imshow('bilateral filter', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-------------------------直方图均衡化-----------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('dark.jpg', 0)
    dst = cv2.equalizeHist(img)
    #彩色图像
    img1 = cv2.imread('dark2.jpg', 1)
    (b,g,r) = cv2.split(img1)
    bh = cv2.equalizeHist(b)
    gh = cv2.equalizeHist(g)
    rh = cv2.equalizeHist(r)
    dst1 = cv2.merge((bh,gh,rh))
    cv2.imshow('orgin gray',img)
    cv2.imshow('orgin color',img1)
    cv2.imshow('equalize gray',dst)
    cv2.imshow('equalize color',dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-----------------------------gamma变换------------------------
def adjust(image,gamma=1.0):
    inv = 1 / gamma
    table = []
    for i in range(256):
        table.append(((i/255) ** inv) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)

flag = 1
# flag = 0
if flag:
    img = cv2.imread('dark2.jpg', 1)
    dst = adjust(img,gamma=1.5)
    cv2.imshow('orgin',img)
    cv2.imshow('gamma',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()