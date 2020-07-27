#----01 图像基础-----
# 图像读写 保存 显示 灰度变换 直方图 通道分离 通道合并
# author：RingoWu
# time：2020.7.18

import cv2
import numpy as np
import matplotlib.pyplot as plt

#--------------------读入图像 显示并保存------------------
# flag = 1
flag = 0
if flag :
    img = cv2.imread('hanxue.jpg', 1)
    cv2.imshow("photo", img)
    k = cv2.waitKey(0) #waiting key
    if k == 27:
        cv2.destroyAllWindows() #按ESC结束显示
    elif k == ord('s'):
        cv2.imwrite('hanxue_1.png', img) #按s进行图像保存
    cv2.destroyAllWindows()

#------------------通道转化，三通道转化为单通道--------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', cv2.IMREAD_UNCHANGED) #读入原始图像
    shape = img.shape
    print(shape)
    if shape[2]==3 or shape[2]==4:
        #彩色转化为灰色
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_gray', img_gray)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#--------------------通道转化， 单通道转化为三通道---------------
# flag = 1
flag = 0
if flag:
    img_gray = cv2.imread('girl_gray.jpg',cv2.IMREAD_UNCHANGED)
    cv2.imshow('img_gray', img_gray)
    shape = img_gray.shape
    print(shape)
    img_three = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    print(img_three.shape)
    cv2.imshow("img_3channel", img_three)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#--------------------图像三通道的分离--------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', cv2.IMREAD_UNCHANGED)
    b,g,r = cv2.split(img)
    #显示三通道的图片
    cv2.imshow('blue', b)
    cv2.imshow('green', g)
    cv2.imshow('red',r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', cv2.IMREAD_UNCHANGED)
    b,g,r = cv2.split(img)
    zeros = np.zeros(img.shape[:2],dtype="uint8") #创建原size的零矩阵
    cv2.imshow("blue",cv2.merge([b,zeros,zeros]))
    cv2.imshow("green",cv2.merge([zeros,g,zeros]))
    cv2.imshow("red",cv2.merge([zeros,zeros,r]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---------------------------图像通道合并与分离---------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', cv2.IMREAD_UNCHANGED)
    b,g,r = cv2.split(img)
    g[:] = 0
    img_merge = cv2.merge([b,g,r])
    cv2.imshow("img_merge", img_merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-------------------------RGB与BGR的转化-------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg',cv2.IMREAD_UNCHANGED)
    img_cv_method = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_numpy_method = img[:,:,::-1] #逆序
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    plt.imshow(img_cv_method)
    plt.subplot(1,3,3)
    plt.imshow(img_numpy_method)

#--------------------------BGR 与 HSV -------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', cv2.IMREAD_UNCHANGED)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img",img)
    cv2.imshow("img_hsv", img_hsv)
    cv2.imshow("img_gray", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-----------------------图像直方图-------------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', cv2.IMREAD_UNCHANGED)
    plt.hist(img.ravel(),256,[0,256]) #3维降到1维 ravel
    plt.show()
    #opencv方法
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

#-------------------------三通道直方图绘制----------------------
flag = 1
# flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', cv2.IMREAD_UNCHANGED)
    color = ("blue", "green", "red")
    for i,color in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.title("hanxue")
        plt.xlabel("Bins")
        plt.ylabel("num of perlex")
        plt.plot(hist, color=color)
        plt.xlim([0, 260])
    plt.show()
