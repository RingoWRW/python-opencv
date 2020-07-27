#----03 阈值分割-----
# 固定阈值、自适应阈值、otsu大津法
# 图像边缘提取
# 区域生长
# 分水岭算法
# author：RingoWu
# time：2020.7.23

import cv2
import numpy as np
import matplotlib.pyplot as plt

#----------------固定阈值--------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('person.png',0)
    ret, dst = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret, dst1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, dst2 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, dst3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, dst4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    images = [img,dst,dst1,dst2,dst3,dst4]
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i],cmap='gray')
        plt.xticks([]),plt.yticks([])
    plt.show()

#------------------自适应阈值---------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('paper2.png', 0)
    ret, dst = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    dst1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,4)
    dst2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)
    images = [img,dst,dst1,dst2]
    title = ['orgin','thresh','mean_thresh','gaussian_thresh']
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(title[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

#---------------------otsu大津法---------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('noisy.png', 0)
    #全局阈值
    ret1,dst = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    #otsu阈值
    ret2,dst1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #先高斯滤波再otsu
    blur = cv2.GaussianBlur(img,(3,3),0)
    ret3,dst2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    images = [img,0,dst,img,0,dst1,blur,0,dst2]
    title = ['orgin','histogram','global',
             'orgin','histogram','otsu',
             'Gassian_blur','histogram','otsu']
    for i in range(3):
        #绘制原图
        plt.subplot(3,3,i*3+1)
        plt.imshow(images[i*3],'gray')
        plt.title(title[i*3])
        plt.xticks([]), plt.yticks([])
        #绘制直方图
        plt.subplot(3,3,i*3+2)
        plt.hist(images[i*3].ravel(),256)
        plt.title(title[i*3+1])
        plt.xticks([]), plt.yticks([])
        #绘制阈值图
        plt.subplot(3,3,(i+1)*3)
        plt.imshow(images[i*3+2],'gray')
        plt.title(title[i*3+2])
        plt.xticks([]), plt.yticks([])
    plt.show()

#----------------------canny边缘检测-------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', 0)
    dst = cv2.Canny(img,50,100,(3,3))  #阈值不同连接的图像就会有所差异
    dst1 = cv2.Canny(img,70,150,(3,3))
    imgs = np.hstack((dst,dst1))
    cv2.imshow('canny',imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#--------------------区域生长-----------------------------
# 效果对于人像来说比较差
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def getx(self):
        return self.x
    def gety(self):
        return self.y
def get_dist(seed_location1,seed_location2):
    l1 = im[seed_location1.x , seed_location1.y]
    l2 = im[seed_location2.x , seed_location2.y]
    count = np.sqrt(np.sum(np.square(l1-l2)))
    return count

# flag = 1
flag = 0
if flag:
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0),
                Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]
    im = cv2.imread("222.jpg")
    #获取高宽
    height = im.shape[0]
    width = im.shape[1]
    img_mark = np.zeros([height, width])  #判断种子生长
    #建立空图像数组
    im_empty = im.copy()
    #保留分割后的色彩
    for i in range(height):
        for j in range(width):
            im_empty[i, j][0] = 0
            im_empty[i, j][1] = 0
            im_empty[i, j][2] = 0
    #纯黑白的分割
    # im_empty = np.zeros([height,width,3])
    cv2.imshow('empty', im_empty)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #获取种子点
    seed_list = []
    seed_list.append(Point(16,16))
    T = 5 #阈值
    class_k = 1 #标记
    while(len(seed_list) > 0):
        seed_tmp = seed_list[0]
        seed_list.pop(0) #删除种子点
        img_mark[seed_tmp.x, seed_tmp.y] = class_k
        #遍历8领域
        for i in range(8):
            tmpx = seed_tmp.x + connects[i].x
            tmpy = seed_tmp.y + connects[i].y

            if(tmpx < 0 or tmpy < 0 or tmpy >= width or tmpx >= height):
                continue
            dist = get_dist(seed_tmp,Point(tmpx,tmpy))
            if(dist < T and img_mark[tmpx,tmpy] == 0):
                im_empty[tmpx,tmpy][0] = im[tmpx,tmpy][0]
                im_empty[tmpx, tmpy][1] = im[tmpx, tmpy][1]
                im_empty[tmpx, tmpy][2] = im[tmpx, tmpy][2]
                img_mark[tmpx,tmpy] = class_k
                seed_list.append(Point(tmpx,tmpy))

    cv2.imshow('orgin',im)
    cv2.imshow('segment',im_empty)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#------------------------分水岭运算----------------------------
flag = 1
# flag = 0
if flag:
    img = cv2.imread('coins.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #阈值分割
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #进行开运算 先腐蚀后膨胀
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow('opening', opening)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #进行膨胀，得到背景区域
    bg = cv2.dilate(opening,kernel,iterations=3)
    cv2.imshow('bg',bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #通过距离变化获取前景区域
    dist = cv2.distanceTransform(bg,cv2.DIST_L2,5)
    ret1, fg = cv2.threshold(opening, 0.1*dist.max(), 255,0)
    cv2.imshow('dist', dist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #背景与前景相减
    fg = np.uint8(fg)
    unknown = bg - fg
    cv2.imshow('unknown', unknown)
    cv2.imshow('fg', fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #连通区域处理
    ret2, mark = cv2.connectedComponents(fg, connectivity=8)
    print(mark.shape,ret2)
    mark += 1
    mark[unknown==255] = 0
    #分水岭
    mark = cv2.watershed(img,mark)

    img[mark == -1] = [0,0,255]
    cv2.imshow('watershed', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

