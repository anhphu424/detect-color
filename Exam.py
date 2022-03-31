import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)  # Capturing Video feed from the Webcam

while True:
    ret, frame = cap.read()  # "frame" is the original video frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting the frame to grayscale

    # Applying a threshold
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Removing the noise
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, kernel, iterations=1)

    # Applying dilation to obtain the sure backgroud
    sure_bg = cv2.dilate(closed, kernel, iterations=2)

    # using Distance transformation
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)

    # Applying another thershold to obtain the centres of the sure foreground
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding the unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Using markers to distinguish foreground and background
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Applying Watershed Algorithm
    markers = cv2.watershed(frame, markers)

    # Obtaining contours and drawing them
    contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            res = cv2.drawContours(frame.copy(), contours, i, color=(0, 255, 0), thickness=2)

    cv2.imshow('Original', frame)  # Displaying Orignal video feed
    cv2.imshow('Contours', res)  # Displaying Video feed with contours

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()