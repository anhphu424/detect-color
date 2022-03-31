import cv2
import time
import numpy as np
import os
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation


def Min_H(pos):
    Min_H.value = pos
Min_H.value = 0

def Min_S(pos):
    Min_S.value = pos
Min_S.value = 99

def Min_V(pos):
    Min_V.value = pos
Min_V.value = 0

def Max_H(pos):
    Max_H.value = pos
Max_H.value = 112

def Max_S(pos):
    Max_S.value = pos
Max_S.value = 255

def Max_V(pos):
    Max_V.value = pos
Max_V.value = 255

cv2.namedWindow("Control")
cv2.createTrackbar("Min H", "Control", 0, 255, Min_H)
cv2.createTrackbar("Min S", "Control", 99, 255, Min_S)
cv2.createTrackbar("Min V", "Control", 0, 255, Min_V)

cv2.createTrackbar("Max H", "Control", 112, 255, Max_H)
cv2.createTrackbar("Max S", "Control", 255, 255, Max_S)
cv2.createTrackbar("Max V", "Control", 255, 255, Max_V)

cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = 1000 / fps

print (fps)
segmentor = SelfiSegmentation()
play = True
delta_time = 0

while True:
    pre_time = time.time()

    if play:
        ret, img = cap.read()

    if img is None:
        img = temp_img
    else:
        temp_img = img
    img_clone = img.copy()
    cv2.imshow("Control", cv2.putText(img_clone.copy(),"%.2f (ms)" %delta_time,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2))
    img_hsv = cv2.cvtColor(img_clone, cv2.COLOR_BGR2HSV)
    lower = np.array([Min_H.value, Min_S.value, Min_V.value])
    upper = np.array([Max_H.value, Max_S.value, Max_V.value])
    mask = cv2.inRange(img_hsv, lower, upper)


    kernel = np.ones((5, 5))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)


    cv2.imshow("Result", mask)
    cv2.imshow("Rsult", img)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q') or key == 27 or 'x' == chr(key & 255):
        exit
cap.release()
cv2.destroyAllWindows()
