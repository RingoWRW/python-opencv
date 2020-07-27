#----02 图像形态学-----
# 图像腐蚀  图像膨胀
# 开运算 闭运算
# 形态学基本梯度 ： 膨胀-腐蚀
# 顶帽和黑帽
# author：RingoWu
# time：2020.7.21

import cv2
import numpy as np
import matplotlib.pyplot as plt

#-----------------------图像腐蚀-----------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('morphology.png')
    kernel = np.ones((3,3),np.uint8)
    dst = cv2.erode(img,kernel,iterations=1)
    cv2.imshow('orgin',img)
    cv2.imshow('erode',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-----------------------图像膨胀-----------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('morphology.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3,))
    dst = cv2.dilate(img,kernel,iterations=1)
    plt.subplot(1,2,1)
    plt.imshow(img),plt.title('orgin')
    plt.xticks([]),plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(dst),plt.title('dilate')
    plt.xticks([]),plt.yticks([])
    plt.show()

#---------------------开运算 先腐蚀后膨胀-------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('open.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    dst = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    plt.subplot(1, 2, 1)
    plt.imshow(img), plt.title('orgin')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(dst), plt.title('open')
    plt.xticks([]), plt.yticks([])
    plt.show()

#----------------------闭运算 先膨胀后腐蚀-------------------------
flag = 1
# flag = 0
if flag:
    img = cv2.imread('morphology.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
    kernel = np.ones((8,8),np.uint8) #kernel大小可以自己根据实验效果改变
    dst = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    plt.subplot(1, 2, 1)
    plt.imshow(img), plt.title('orgin')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(dst), plt.title('close')
    plt.xticks([]), plt.yticks([])
    plt.show()

#------------------形态学梯度 膨胀-腐蚀----------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('morphology.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.ones((3,3), np.uint8)
    dst = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
    plt.subplot(1, 2, 1)
    plt.imshow(img), plt.title('orgin')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(dst), plt.title('gradient')
    plt.xticks([]), plt.yticks([])
    plt.show()

#-------------------顶帽 原图与开运算（突出原图亮区域）----------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('morphology.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.ones((9,9), np.uint8)
    dst = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
    plt.subplot(1, 2, 1)
    plt.imshow(img), plt.title('orgin')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(dst), plt.title('tophat')
    plt.xticks([]), plt.yticks([])
    plt.show()

#-------------------黑帽 原图与闭运算（突出原图暗区域）----------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('morphology.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5,5), np.uint8)
    dst = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
    plt.subplot(1, 2, 1)
    plt.imshow(img), plt.title('orgin')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(dst), plt.title('blackhat')
    plt.xticks([]), plt.yticks([])
    plt.show()