from msilib.schema import Binary
import cv2
from matplotlib.pyplot import contour
import numpy as np

height = 720
width = 1280

frame_width = 640
frame_height = 480

img = cv2.imread('b.png',1)
#img = cv2.resize(img, (width, height))
new_img = img.copy()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray = cv2.resize(gray, (width, height))
cv2.imshow('gray', gray)

ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_OTSU)
#binary = cv2.resize(binary, (width, height))
cv2.imshow('Binary', binary)

inverted = ~binary
#inverted = cv2.resize(inverted, (width, height))
cv2.imshow('Inverted', inverted)

contour, hierarchy = cv2.findContours(inverted,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
drawcontour = cv2.drawContours(img, contour, -1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow('DrawContour', drawcontour)

cv2.waitKey(0)
cv2.destroyAllWindows()
#cap.release()