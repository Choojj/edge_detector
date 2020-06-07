import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img_source = cv2.imread('C:/Users/HSWB/Desktop/edge_detector/sample/modified_03.png', cv2.IMREAD_GRAYSCALE)

ret,img_result1 = cv2.threshold(img_source, 127, 255, cv2.THRESH_BINARY)

ret, img_result2 = cv2.threshold(img_source, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img_blur = cv2.GaussianBlur(img_source, (3, 3), 0)
ret, img_result3 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



cv2.imshow("SOURCE", img_source)
cv2.imshow("THRESH_BINARY", img_result1)
cv2.imshow("THRESH_OTSU", img_result2)
cv2.imshow("THRESH_OTSU + Gaussian filtering", img_result3)

cv2.waitKey(0)
cv2.destroyAllWindows()