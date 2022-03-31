import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

height = 480
width = 640

frame_width = 640
frame_height = 480
video_capture = cv2.VideoCapture(1)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# using flag value
video_capture.set(3, frame_width)
video_capture.set(4, frame_height)

while(video_capture.isOpened()):
    ret, frame = video_capture.read()

    #frame = cv2.resize(frame,(int(frame.shape[1]/2), int(frame.shape[0]/2)))
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #low = np.array([0, 76, 212])
    #upper = np.array([4, 255, 255])
    #mask = cv2.inRange(hsv, low, upper)
    img = frame
    img = cv2.resize(img, (width, height))
    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_back = frame

    muy = 0
    arr_green = np.zeros((361,), np.uint32)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            k = HSV[i, j]
            muy = muy + k[0]
            arr_green[k[0]] = arr_green[k[0]] + 1
    hist = arr_green.ravel()

    muy = muy / (HSV.shape[1] * HSV.shape[0])

    w, h = HSV.shape[:2]
    temp = 0.0
    for x in range(0, HSV.shape[0]):
        for y in range(0, HSV.shape[1]):
            k = HSV[x, y]
            temp = temp + (muy - k[0]) ** 2
    sigma = math.sqrt(temp / w / h)

    img_mask = np.zeros((w, h), np.uint16)
    for x in range(0, HSV.shape[0]):
        for y in range(0, HSV.shape[1]):
            k = HSV[x, y]
            if (k[0] > muy - sigma) and (k[0] < muy + sigma):
                img_mask[x, y] = 1

    img_result = img.copy()
    for x in range(0, HSV.shape[0]):
        for y in range(0, HSV.shape[1]):
            if img_mask[x, y] == 1:
                img_result[x, y] = img_back[x, y]

    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img_result, kernel, iterations=3)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=3)
    gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=img_result, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)
    cv2.imshow("Cam", img_result)
   # cv2.imshow("Thresh", HSV)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

video_capture.release()
cv2.destroyAllWindows()