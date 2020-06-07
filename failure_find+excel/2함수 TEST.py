import cv2
import numpy as np

def extraction(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    # blurred = cv2.GaussianBlur(image, (5, 5), 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

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

        if area > 120000 and area < 1400000:
            dst = image.copy()
            dst = image[top - 10:top + height + 10, left - 10:left + width + 10]

    return dst

def diff_clip(imageA, imageB):
    height, width = 500, 1000
    blank_image = np.zeros((height, width, 1), np.uint8)
    blank_imageA = cv2.bitwise_not(blank_image)
    blank_imageB = cv2.bitwise_not(blank_image)

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    ret, resultA = cv2.threshold(grayA, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, resultB = cv2.threshold(grayB, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    resultA_y, resultA_x = resultA.shape
    resultB_y, resultB_x = resultB.shape

    blank_imageA[height - resultA_y:height, 0:resultA_x] = resultA
    blank_imageB[height - resultB_y:height, 0:resultB_x] = resultB

    absdiff = cv2.absdiff(blank_imageA, blank_imageB)
    return absdiff, blank_imageA, blank_imageB

def diff_feature(diff_img, print_img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_img)

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
        cv2.rectangle(print_img, (left, top), (left + width, top + height), 125, 2)

path1 = "C:/Users/HSWB/Desktop/edge_detector/data/original_01.png"
path2 = "C:/Users/HSWB/Desktop/edge_detector/data/modified_14.png"

clip1 = extraction(path1)
clip2 = extraction(path2)

absdiff, blankA, blankB = diff_clip(clip1, clip2)

diff_feature(absdiff, blankA)

cv2.imshow("result1", blankA)
cv2.imshow("result2", blankB)
cv2.imshow("result3", absdiff)
cv2.waitKey(0)
cv2.destroyAllWindows()