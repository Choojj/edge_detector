import cv2
import numpy as np

path = "C:/Users/HSWB/Desktop/edge_detector/data1/original_01.png"

image = cv2.imread(path, cv2.IMREAD_COLOR)
image = image[225:919, 82:1816]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edge = cv2.Canny(gray, 50, 100)

edge = cv2.bitwise_not(edge)

contours = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(edge, contours[0], -1, (0, 0, 0), 1, cv2.LINE_8)

contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if (i > 0):
        # epsilon = 0.005 * cv2.arcLength(contours[i], True)
        # approx = cv2.approxPolyDP(contours[i], epsilon, True)
        # 컨투어 근사치를 사용하기때문에 오차가 커짐 -> 근사치를 사용하지않고 그대로 다각형roi를 사용해 가져온후 하면 될것같다!

        for j in range(len(approx)-1):
            print(approx[j][0], end = " ")
            cv2.line(image, tuple(approx[j][0].tolist()), tuple(approx[j+1][0].tolist()), (0, 0, 255), 1)
            cv2.imshow("a", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print()

# cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

cv2.imshow("a", image)
cv2.waitKey(0)
cv2.destroyAllWindows()