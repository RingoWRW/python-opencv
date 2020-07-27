#----04 图像特征-----
# Hog特征
# Harris  && SIFT
# 纹理特征：LBP特征
# 模板匹配
# 人脸检测
# author：RingoWu
# time：2020.7.27

import cv2
import matplotlib.pyplot as plt
import numpy as np

#-------------------------HoG特征---------------------------

def is_inside(o, i):
    """
    :param o:
    :param i:
    :return: 是否o在i内
    """
    ox,oy,ow,oh = o
    ix,iy,iw,ih = i
    return ox > ix and oy > iy and ox+ow < ix+iw and oy+oh < iy+ih

def draw_person(image,person):
    x,y,w,h = person
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,200,200),2)

# flag = 1
flag = 0
if flag:
    img = cv2.imread('people.jpg')
    hog = cv2.HOGDescriptor()  #启动检测器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  #检测器类型为人
    found, w = hog.detectMultiScale(img,0.1,(1,1))
    print(found)   #位置
    print(w)

    found_filter = []
    #对有包含关系的框进行剔除
    for x,y in enumerate(found):
        for x1,y1 in enumerate(found):
            if x != x1 and is_inside(y, y1):
                break
        else:
            found_filter.append(y)
            print(found_filter)
    for person in found_filter:
        draw_person(img,person)
    cv2.imshow("people dectet", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---------------------------harris---------------------------

# flag = 1
flag = 0
if flag:
    img = cv2.imread('harris.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray,2,3,0.03)
    #膨胀显示更清楚
    dst = cv2.dilate(dst,None)
    #角点显示
    img[dst>0.01*dst.max()] = [0,200,200]
    cv2.imshow("harris",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #sift = cv2.xfeature2d.SIFT_create()

#------------------------LBP特征------------------------

def LBP(src):
    """
    :param src: 灰度图像
    :return:
    """
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    lbp_value = np.zeros((1,8), dtype=np.uint8)
    neig = np.zeros((1,8), dtype=np.uint8)

    for x in range(1, width-1):
        for y in range(1, height-1):

            neig[0,0] = src[y-1,x-1]
            neig[0,1] = src[y-1,x]
            neig[0,2] = src[y-1,x+1]
            neig[0,3] = src[y,x-1]
            neig[0,4] = src[y,x+1]
            neig[0,5] = src[y+1,x-1]
            neig[0,6] = src[y+1,x]
            neig[0,7] = src[y+1,x+1]
            center = src[y,x]

            for i in range(8):
                if neig[0,i] > center:
                    lbp_value[0,i] = 1
                else:
                    lbp_value[0,i] = 0

            lbp = lbp_value[0,0]*1 + lbp_value[0,1]*2 + lbp_value[0,2]*4 + lbp_value[0,3]*8 + lbp_value[0,4]*16
            + lbp_value[0,5]*32 + lbp_value[0,6]*64 + lbp_value[0,7]*128

            dst[y,x] = lbp

    return dst

# flag = 1
flag = 0
if flag:
    img = cv2.imread('people.jpg', 0)
    cv2.imshow("orgin", img)
    cv2.waitKey(0)
    new = LBP(img)
    cv2.imshow("LBP", new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-----------------------模板匹配-------------------------
def template(temp,src):
    method = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
    th, tw = temp.shape[:2]
    for md in method:
        result = cv2.matchTemplate(src, temp, md)
        min_dist, max_dist, min_loc, max_loc = cv2.minMaxLoc(result)
        print(min_dist, max_dist, min_loc, max_loc)
        if md == cv2.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)
        cv2.rectangle(src, tl, br, (0,200,200),2)
        cv2.namedWindow("match-" + np.str(md), cv2.WINDOW_AUTOSIZE)
        cv2.imshow("match-"+np.str(md), src)

# flag = 1
flag = 0
if flag:
    src = cv2.imread('target1.jpg')
    temp = cv2.imread('sample2.jpg')
    template(temp, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#-------------------------人脸识别----------------------------
