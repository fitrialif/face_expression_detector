import numpy as np
import cv2

while True:
    a = cv2.imread('test_img.jpg')
    a = cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    a = cv2.resize(a, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("",a)
    cv2.waitKey(1)