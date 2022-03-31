import cv2 
import numpy as np
import math

last = True

height = 480
width = 640

frame_width = 640
frame_height = 480
fps = 30.0

video_capture = cv2.VideoCapture(1)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# using flag value
video_capture.set(3, frame_width)
video_capture.set(4, frame_height)

while(video_capture.isOpened()):
    ret, frame = video_capture.read()

    #img = cv2.imread('3.jpg')
    img = frame
    img = cv2.resize(img, (width,height))
    img_back = cv2.imread('backgruond.jpg')

    img_back = cv2.resize(img_back, (width,height))
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    muy = 0
    arr_green = np.zeros((361,), np.uint32)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            k = img_HSV[i,j]
            muy = muy + k[0]
            arr_green[k[0]] = arr_green[k[0]] + 1 
    hist = arr_green.ravel()

    muy = muy/(img_HSV.shape[1]*img_HSV.shape[0])

    w, h = img_HSV.shape[:2]
    temp = 0.0
    for x in range(0, img_HSV.shape[0]):
        for y in range(0, img_HSV.shape[1]):
            k = img_HSV[x,y]
            temp = temp + (muy - k[0]) ** 2
    sigma = math.sqrt(temp/w/h)

    img_mask = np.zeros((w,h),np.uint16)
    for x in range(0, img_HSV.shape[0]):
        for y in range(0, img_HSV.shape[1]):
            k = img_HSV[x,y]
            if (k[0]> muy - sigma) and (k[0] < muy + sigma):
                img_mask[x,y] = 1

    img_result = img.copy()
    for x in range(0, img_HSV.shape[0]):
        for y in range(0, img_HSV.shape[1]):
            if img_mask[x,y] == 1:
                img_result[x,y] = img_back[x,y]

    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img_result, kernel, iterations=3)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=3)

    cv2.imshow('frame', img_result)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q') or key == 27 or 'x' == chr(key & 255):
        exit