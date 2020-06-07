import cv2
import numpy as np

# 문제가 생김 -> 차영상을 구해서 도형으로 구분했는데 보정하고 나서 컨투어 외부 픽셀을 바꾸는 법을 모르겠어서(알지만 시간이 오래걸림) 차영상을 구할수가 없음
# 좌표로만 비교해야 할것 같음
# cv2.connectedComponentsWithStats 함수로 구했던것들 모두 cv2.findContours로 구하도록 바꿔야 할듯
# 일단 잘나오기는 하는데 추출해야하는 컨투어 잘안됨 컨투어찾아서 꼭지점수로 구분해야 할듯!

def extraction(path): # 원본에서 roi 추출
    image = path

    # blurred = cv2.GaussianBlur(image, (5, 5), 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    edge = cv2.Canny(gray, 50, 150)

    edge = cv2.bitwise_not(edge)

    contours = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, contours[0], -1, (0, 0, 0), 1)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge)
    for i in range(nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        center_x = int(centroids[i, 0])
        center_y = int(centroids[i, 1]) 
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if 80000 < area < 1000000 and 500 < center_x < 1500: # 임시로 도형 센터 값으로 구분
            print(i, centroids[i], area)
            dst = image.copy()
            dst = image[top - 10:top + height + 10, left - 10:left + width + 10]

    return dst

def diff_clip(imageA, imageB): # roi에서 차영상 추출
    height, width = 600, 1000
    blank_image = np.zeros((height, width, 3), np.uint8)
    blank_imageA = cv2.bitwise_not(blank_image)
    blank_imageB = cv2.bitwise_not(blank_image)

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    blank_imageA = cv2.cvtColor(blank_imageA, cv2.COLOR_BGR2GRAY)
    blank_imageB = cv2.cvtColor(blank_imageB, cv2.COLOR_BGR2GRAY)

    ret, resultA = cv2.threshold(grayA, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, resultB = cv2.threshold(grayB, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    resultA_y, resultA_x = resultA.shape
    resultB_y, resultB_x = resultB.shape

    blank_imageA[height - resultA_y:height, 0:resultA_x] = resultA
    blank_imageB[height - resultB_y:height, 0:resultB_x] = resultB

    absdiff = cv2.absdiff(blank_imageA, blank_imageB)
    return absdiff, blank_imageA, blank_imageB


path = "C:/Users/HSWB/Desktop/edge_detector/data1/rotation_01_01.png"

origianl_point = []

image = cv2.imread(path, cv2.IMREAD_COLOR)
cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("result", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge = cv2.Canny(gray, 50, 100)
cv2.imshow("result", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge = cv2.bitwise_not(edge)
cv2.imshow("result", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(edge, contours[0], -1, (0, 0, 0), 1, cv2.LINE_8)
cv2.imshow("result", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in enumerate(contours):
    epsilon = 0.005 * cv2.arcLength(cnt[1], True)
    approx = cv2.approxPolyDP(cnt[1], epsilon, True)

    if (80000 < cv2.contourArea(cnt[1]) < 1000000):
        print(f"컨투어 계층관계 = {cnt[0]} {cv2.contourArea(cnt[1])} {hierarchy[0][cnt[0]]}")
        if (hierarchy[0][cnt[0]][2] == -1 and len(approx) >= 12):
            print(len(approx), end = " ")
            print()
            Max = Max2 = -float("inf")
            Min = Min2 = float("inf")
            Max_num = Max2_num = 0
            Min_num = Min2_num = 0
            for i in range(len(approx)):
                print(i, approx[i][0], end = " ")
                if (approx[i][0][0] >= Max):
                    Max2 = Max
                    Max2_num = Max_num
                    Max = approx[i][0][0]
                    Max_num = i
                elif (Max2 < approx[i][0][0] < Max):
                    Max2 = approx[i][0][0]
                    Max2_num = i
                if (approx[i][0][0] <= Min):
                    Min2 = Min
                    Min2_num = Min_num
                    Min = approx[i][0][0]
                    Min_num = i
                elif (Min2 > approx[i][0][0] > Min):
                    Min2 = approx[i][0][0]
                    Min2_num = i
            print()
            print(f"M = {Max, Max_num}, M2 = {Max2, Max2_num}, m = {Min, Min_num}, m2 = {Min2, Min2_num}") # x좌표 최대인점 최소인점 2개 더구해서 기준점으로 사용
            if (approx[Min_num][0][1] > approx[Min2_num][0][1]):
                origianl_point.append([])
                origianl_point[0].append(approx[Min2_num][0][0])
                origianl_point[0].append(approx[Min2_num][0][1])
                origianl_point.append([])
                origianl_point[1].append(approx[Min_num][0][0])
                origianl_point[1].append(approx[Min_num][0][1])
                if (approx[Max_num][0][1] > approx[Max2_num][0][1]):
                    origianl_point.append([])
                    origianl_point[2].append(approx[Max2_num][0][0])
                    origianl_point[2].append(approx[Max2_num][0][1])
                    origianl_point.append([])
                    origianl_point[3].append(approx[Max_num][0][0])
                    origianl_point[3].append(approx[Max_num][0][1])
                else:
                    origianl_point.append([])
                    origianl_point[2].append(approx[Max_num][0][0])
                    origianl_point[2].append(approx[Max_num][0][1])
                    origianl_point.append([])
                    origianl_point[3].append(approx[Max2_num][0][0])
                    origianl_point[3].append(approx[Max2_num][0][1])
            else:
                origianl_point.append([])
                origianl_point[0].append(approx[Min_num][0][0])
                origianl_point[0].append(approx[Min_num][0][1])
                origianl_point.append([])
                origianl_point[1].append(approx[Min2_num][0][0])
                origianl_point[1].append(approx[Min2_num][0][1])
                if (approx[Max_num][0][1] > approx[Max2_num][0][1]):
                    origianl_point.append([])
                    origianl_point[2].append(approx[Max2_num][0][0])
                    origianl_point[2].append(approx[Max2_num][0][1])
                    origianl_point.append([])
                    origianl_point[3].append(approx[Max_num][0][0])
                    origianl_point[3].append(approx[Max_num][0][1])
                else:
                    origianl_point.append([])
                    origianl_point[2].append(approx[Max_num][0][0])
                    origianl_point[2].append(approx[Max_num][0][1])
                    origianl_point.append([])
                    origianl_point[3].append(approx[Max2_num][0][0])
                    origianl_point[3].append(approx[Max2_num][0][1])
            print(origianl_point)

img = cv2.imread(path)
h, w = img.shape[:2]

pts1 = np.float32(origianl_point)
pts2 = np.float32([[637, 615], [637, 741], [1282, 615], [1282, 741]])

M = cv2.getPerspectiveTransform(pts1, pts2)

img2 = cv2.warpPerspective(img, M, (w, h))

cv2.imshow("B", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

clip = extraction(img2)

cv2.imshow("B", clip)
cv2.waitKey(0)
cv2.destroyAllWindows()