import numpy as np
import cv2
from matplotlib import pyplot as plt

a = []

for i in range(100):
    line = (0, 0)
    a.append(line)
    
src = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/data1/original_01.png", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

corners = cv2.goodFeaturesToTrack(otsu, 100, 0.05, 10, blockSize = 3, useHarrisDetector = True, k = 0.03)

j = 0

for i in corners:
    x, y = i.ravel()
    a[j] = x, y
    cv2.circle(src, (x, y), 3, 255, -1)
    j += 1

    text = "%d,%d" %(x, y)
    cv2.putText(src, text, (x, y), 2, 0.3, (0, 0, 0))

a.sort()
print(a)

cv2.imshow("src", src)
cv2.imshow("otsu", otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
3개 -> 0.2mm 0.4mm 0.8mm 1.0mm 2.0mm 3.0mm
       6 6 | 9 10 | 16 16 | 18 18 | 33 33 | 48 48 수직차이
       0 0   0 0    0  0    0  0    0  0    0 0   수평차이

7개 -> 5 5 | 2 2 | 6 5 | 2 2 | 2 2 | 5 5 | 2 2 수직차이
       0 0   1 0   0 0   0 0   0 0   0 0   0 0 수평차이
'''