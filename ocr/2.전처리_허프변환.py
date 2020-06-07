import cv2
import numpy as np

image = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/ocr/data/modified_01.png")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.bitwise_not(image)

edge = cv2.Canny(image, 50, 200, apertureSize = 3)
cv2.imshow("edge", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

rho = 1 # r의 범위 (0 ~ 1)
theta = np.pi / 360 # 세타의 범위 (0 ~ 180)
threshold = 200 # 만나는 점의 기준 / 정확도
minLineLength = 100 # 선의 최소 길이
maxLineGap = 0 # 선 간격

lines = cv2.HoughLinesP(edge, rho, theta, threshold, minLineLength, maxLineGap)
for i in range(len(lines)):
    cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 0), 5)

cv2.imshow('img1', image)
cv2.waitKey(0)
cv2.destroyAllWindows()