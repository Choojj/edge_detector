import numpy as np
import cv2
from matplotlib import pyplot as plt

a = []

for i in range(100):
    line = (0, 0)
    a.append(line)
    
src = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/sample/modified_13.png", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret1, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
ret2, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 100, blockSize = 5, useHarrisDetector = True, k = 0.03)

j = 0

for i in corners:
    x, y = i.ravel()
    a[j] = x, y
    cv2.circle(src, (x, y), 2, 255, -1)
    j += 1

cv2.imshow("src", src)
cv2.imshow("gray", gray)
cv2.imshow("otsu", otsu)
cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

#첫점과 나머지 3점의 거리차로 나머지 점 위치 추측