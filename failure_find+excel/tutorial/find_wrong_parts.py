# 0.03mm는 가끔씩 구분 0.05mm부터 확실히 구분

import cv2
import numpy as np

def extraction(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) 

    edge = cv2.Canny(gray, 50, 150)

    edge = cv2.bitwise_not(edge)

    contours = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, contours[0], -1, (0, 0, 0), 1)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge)
    for i in range(nlabels):
        if i < 2:
            continue

        area = stats[i, cv2.CC_STAT_AREA]
        center_x = int(centroids[i, 0])
        center_y = int(centroids[i, 1]) 
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if area > 130000 and area < 1400000:
            dst = image.copy()
            dst = image[top - 10:top + height + 10, left - 10:left + width + 10]

    return dst

def diff_clip(imageA_path, imageB_path):
    imageA = extraction(imageA_path)
    imageB = extraction(imageB_path)

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    ret, resultA = cv2.threshold(grayA, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, resultB = cv2.threshold(grayB, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    absdiff = cv2.absdiff(resultA, resultB)
    return absdiff

pathA = "C:/Users/HSWB/Desktop/edge_detector/sample/original_01.png"
pathB = "C:/Users/HSWB/Desktop/edge_detector/sample/modified_01.png"

absdiff = diff_clip(pathA, pathB)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(absdiff)

for i in range(nlabels):
    if i < 1:
            continue

    area = stats[i, cv2.CC_STAT_AREA]
    center_x = int(centroids[i, 0])
    center_y = int(centroids[i, 1])
    left = stats[i, cv2.CC_STAT_LEFT]
    top = stats[i, cv2.CC_STAT_TOP]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    print(left, top, left + width, top + height)
    cv2.rectangle(absdiff, (left, top), (left + width, top + height), 125, 1)

cv2.imshow("result1", absdiff)
cv2.waitKey(0)
cv2.destroyAllWindows()