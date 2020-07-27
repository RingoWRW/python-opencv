#----02 图像简单操作-----
# 画线 画圆 画矩阵 多边形 椭圆 添加字符
# 图像平移 图像缩放 图像旋转  仿射变换
# 透视变换
# 文档校正（小练习）
# author：RingoWu
# time：2020.7.20

import cv2
import numpy as np
import matplotlib.pyplot as plt

#---------------画直线 圆 矩阵 椭圆 多边形 加字---------------------
# flag = 1
flag = 0
if flag:
    img = np.zeros((512,512,3), dtype=np.uint8)
    cv2.namedWindow("example one")

    cv2.line(img,(20,20),(200,200),(0,0,255),3)
    cv2.circle(img,(50,50),30,(0,255,255),-1)
    cv2.rectangle(img,(250,250),(500,500),(255,0,0),3)
    cv2.ellipse(img,(20,20),(10,20),0,0,360,(200,200,200),3)
    #定义四个点
    ptr = np.array([[50,10],[100,10],[50,40],[100,40]])
    ptr = ptr.reshape((-1,1,2))
    cv2.polylines(img,[ptr],True,(100,100,100),3)
    #加字
    font = cv2.FONT_HERSHEY_DUPLEX #字体样式
    cv2.putText(img,"welcome to opencv",(0,400),font,1,(100,0,100),2,cv2.LINE_AA)
    cv2.imshow("example one", img)
    cv2.waitKey(0)
    cv2.destroyWindow("example one")

#------------------图像平移 ------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread("hanxue.jpg", 1)
    #M矩阵
    H = np.float32([[1,0,25],[0,1,50]])
    raws,cols = img.shape[:2]
    dst = cv2.warpAffine(img,H,(cols,raws))
    cv2.imshow("original picture", img)
    cv2.imshow("move picture", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#------------------图像缩放------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', 1)
    #放大
    dst = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    #缩小
    heights, weights = img.shape[:2]
    dst1 = cv2.resize(img,(int(0.5*heights),int(0.5*weights)),interpolation=cv2.INTER_LINEAR)
    cv2.imshow('orgin', img)
    cv2.imshow('enlarge', dst)
    cv2.imshow('shrink', dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-----------------图像旋转---------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', 1)
    height, weight = img.shape[:2]
    M = cv2.getRotationMatrix2D((weight/2,height/2),50,-1)
    print(M)
    dst = cv2.warpAffine(img,M,(weight,height))
    dst1 = cv2.warpAffine(img,M,(weight*2,height*2),borderValue=(0,255,255))
    cv2.imshow('orgin',img)
    cv2.imshow('无填充',dst)
    cv2.imshow('有填充',dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-------------------图像仿射变换-------------------------
# flag = 1
flag = 0
if flag:
    #读入图片--定义变换矩阵---进行warp---显示
    img = cv2.imread('hanxue.jpg', 1)
    weight,height = img.shape[:2]
    pos1 = np.float32([[10,20],[100,200],[150,200]])
    pos2 = np.float32([[10,50],[140,210],[160,220]])
    M = cv2.getAffineTransform(pos1,pos2)
    print(M)
    dst = cv2.warpAffine(img,M,(int(height*1.5),int(weight*1.5)))
    cv2.imshow('orgin',img)
    cv2.imshow('affine',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-------------------图像透视变换---------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('hanxue.jpg', 1)
    raw, col = img.shape[:2]
    pos1 = np.float32([[114,91],[200,20],[130,50],[187,200]])
    pos2 = np.float32([[50,20],[40,50],[70,108],[110,301]])
    M = cv2.getPerspectiveTransform(pos1,pos2)
    print(M)
    dst = cv2.warpPerspective(img,M,(col,raw))
    cv2.imshow('orgin',img)
    cv2.imshow('PerspectiveTransform', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---------------------文档校正（二维码扫码）------------------------------
# flag = 1
flag = 0
if flag:
    img = cv2.imread('paper.png', 1)
    #获取大小
    raw,col = img.shape[:2]
    #去噪
    img_gassian = cv2.GaussianBlur(img,(3,3),0)
    #灰度处理
    img_gray = cv2.cvtColor(img_gassian,cv2.COLOR_BGR2GRAY)
    #边缘检测
    img_edge = cv2.Canny(img_gray,50,250,apertureSize=3)
    cv2.imwrite('canny_image.jpg',img_edge)
    #霍夫变换得到纸边缘
    lines = cv2.HoughLinesP(img_edge,1,np.pi/180,50,minLineLength=90,maxLineGap=10)
    print(lines.shape,'\n',lines)  #打印出来找四个顶点
    #获取M矩阵
    pos1 = np.float32([[114, 82], [287, 156], [8, 322], [216, 333]])
    pos2 = np.float32([[0,0],[210,0],[0,290],[210,290]])
    M = cv2.getPerspectiveTransform(pos1,pos2)
    print(M)
    #结果图片
    dst = cv2.warpPerspective(img,M,(212,310)) #谨慎设置图像大小
    #显示结果
    cv2.imshow('orgin',img)
    cv2.imshow('gassion',img_gassian)
    cv2.imshow('edge',img_edge)
    cv2.imshow('results', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---------------------图像操作总结小实验----------------------------
flag = 1
# flag = 0
if flag:
    #读取读片并转换色彩通道
    img = cv2.imread('hanxue.jpg', 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #放大 缩小
    img_enlarge = cv2.resize(img,None,fx=1.2,fy=1.2)
    img_shrink = cv2.resize(img,None,fx=0.8,fy=0.9)
    #平移
    M = np.float32([[0,1,20],[1,0,50]])
    row,col = img.shape[:2]
    img_move = cv2.warpAffine(img,M,(col,row))
    #旋转
    M = cv2.getRotationMatrix2D((col/2,row/2),45,-1)
    img_ro = cv2.warpAffine(img,M,(col,row))
    #翻转
    img_x = cv2.flip(img,0) #x轴翻转
    img_y = cv2.flip(img,1) #y轴翻转
    #仿射变换
    pos1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pos2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pos1,pos2)
    img_affine = cv2.warpAffine(img,M,(col,row))
    #透视变换
    pos1 = np.float32([[56, 65], [238, 52], [28, 237], [239, 240]])
    pos2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
    M = cv2.getPerspectiveTransform(pos1,pos2)
    img_perspec = cv2.warpPerspective(img,M,(col,row))
    #显示
    titles = ['orgin','enlarge','shrink','move','rotation','flip_x','flip_y','affine','perspective']
    image = [img,img_enlarge,img_shrink,img_move,img_ro,img_x,img_y,img_affine,img_perspec]
    for i in range(len(titles)):
        plt.subplot(3,3,i+1)
        plt.imshow(image[i])
        plt.title(titles[i])
    plt.show()
