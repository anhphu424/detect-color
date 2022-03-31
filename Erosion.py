import cv2
import numpy as np

img = cv2.imread('img.jpg',1)

kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(img, kernel, iterations=3)
img_erosion = cv2.erode(img_dilation, kernel, iterations=3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)

#for x in range(0,15):
    #img1 = filtererosion(img1,5)

cv2.imshow('img1', img_erosion)
cv2.imshow('img', img)

'''    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            a = img[x - 1:x + 2, y - 1:y + 2]
            d = sum(a / 255)
            print(a) '''


cv2.waitKey(0)
cv2.destroyAllWindows()