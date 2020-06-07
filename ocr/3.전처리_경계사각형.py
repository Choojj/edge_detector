import cv2
import numpy as np

image = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/ocr/data/modified_01.png")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

invert = cv2.bitwise_not(image)
cv2.imshow("invert", invert)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge = cv2.Canny(invert, 50, 200, apertureSize = 3)
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
    cv2.line(edge, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 0), 10)

cv2.imshow('line remove', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

dilation = cv2.dilate(edge, kernel, iterations = 10)
cv2.imshow("dilation", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    box = np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    print(rect)
    print(box)
    cv2.drawContours(image, [box], -1, (0, 0, 255), 2)
cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()