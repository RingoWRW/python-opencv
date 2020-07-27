#----05 视频处理-----
# 帧差法
# 光流法 && 背景减除法
# author：RingoWu
# time：2020.7.27

import cv2
import matplotlib.pyplot as plt
import numpy as np

#------------------帧差法----------------------------
# flag = 1
flag = 0
if flag:
    file = 'move_detect.flv'
    cam = cv2.VideoCapture(file)
    #视频文件参数设置
    video_fps = 12
    fourcce = cv2.VideoWriter_fourcc('M','P','4','2')
    out1 = cv2.VideoWriter('v1.avi', fourcce, video_fps, (500,400))
    out2 = cv2.VideoWriter('v2.avi', fourcce, video_fps, (500, 400))

    #初始化帧
    last_frame = None
    #遍历所有帧
    while cam.isOpened():
        (ret, frame) = cam.read()
        if not ret:
            break
        #调整帧的大小
        frame = cv2.resize(frame, (500,400), interpolation=cv2.INTER_CUBIC)
        if last_frame is None:
            last_frame = frame
            continue
        #计算当前帧和前帧的不同
        frame_delta = cv2.absdiff(last_frame, frame)
        #当前帧设置为下一帧的前帧
        last_frame = frame.copy()
        #结果转换为灰度图
        Delta = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
        #图像二值化
        Delta = cv2.threshold(Delta, 25, 255, cv2.THRESH_BINARY)[1]

        # 阀值图像上的轮廓位置
        cnts, hierarchy = cv2.findContours(Delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历轮廓
        for c in cnts:
            # 忽略小轮廓，排除误差
            if cv2.contourArea(c) < 300:
                continue

                # 计算轮廓的边界框，在当前帧中画出该框
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 显示当前帧
        cv2.imshow("frame", frame)
        cv2.imshow("frameDelta", frame_delta)
        cv2.imshow("thresh", Delta)

        # 保存视频
        out1.write(frame)
        out2.write(frame_delta)

        # 如果q键被按下，跳出循环
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            # 清理资源并关闭打开的窗口
    out1.release()
    out2.release()
    cam.release()
    cv2.destroyAllWindows()

#-------------------------光流法----------------------
# flag = 1
flag = 0
if flag:
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cap = cv2.VideoCapture("move_detect.flv")
    frame1 = cap.read()[1]
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # 视频文件输出参数设置
    out_fps = 12.0  # 输出文件的帧率
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    sizes = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out1 = cv2.VideoWriter('E:/video/v6.avi', fourcc, out_fps, sizes)
    out2 = cv2.VideoWriter('E:/video/v8.avi', fourcc, out_fps, sizes)

    while True:
        (ret, frame2) = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        draw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        draw = cv2.morphologyEx(draw, cv2.MORPH_OPEN, kernel)
        draw = cv2.threshold(draw, 25, 255, cv2.THRESH_BINARY)[1]

        contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imshow('frame2', bgr)

        cv2.imshow('draw', draw)
        cv2.imshow('frame1', frame2)
        out1.write(bgr)
        out2.write(frame2)

        k = cv2.waitKey(20) & 0xff
        if k == 27 or k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', bgr)
        prvs = next

    out1.release()
    out2.release()
    cap.release()
    cv2.destroyAllWindows()

#------------------------背景法--------------------
flag = 1
# flag = 0
if flag:
    cap = cv2.VideoCapture('move_detect.flv')

    # create the subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=100, detectShadows=False)


    def getPerson(image, opt=1):

        # get the front mask
        mask = fgbg.apply(frame)

        # eliminate the noise
        line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)
        cv2.imshow("mask", mask)

        # find the max area contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in range(len(contours)):
            area = cv2.contourArea(contours[c])
            if area < 150:
                continue
            rect = cv2.minAreaRect(contours[c])
            cv2.ellipse(image, rect, (0, 255, 0), 2, 8)
            cv2.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
        return image, mask


    while True:
        ret, frame = cap.read()
        cv2.imshow('input', frame)
        result, m_ = getPerson(frame)
        cv2.imshow('result', result)
        k = cv2.waitKey(20) & 0xff
        if k == 27:
            cv2.imwrite("result.png", result)
            cv2.imwrite("mask.png", m_)

            break
    cap.release()
    cv2.destroyAllWindows()