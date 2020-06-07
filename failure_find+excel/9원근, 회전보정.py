import cv2
import numpy as np

path = "C:/Users/HSWB/Desktop/edge_detector/data1/rotation_08.png"

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
            print(f"M = {Max, Max_num}, M2 = {Max2, Max2_num}, m = {Min, Min_num}, m2 = {Min2, Min2_num}") # x좌표 최대, 최소이면서 가장 아래에 있는 점 2개씩 더구해서 기준점으로 사용
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
pts2 = np.float32([[637, 615], [637, 741], [1282, 615], [1282, 741]]) # 원래 기준점 4개와 현재 기준점 4개 정의

M = cv2.getPerspectiveTransform(pts1, pts2) # 평면 원근 투영변환 매트릭스 생성

img2 = cv2.warpPerspective(img, M, (w, h)) # 변환

cv2.imshow("A", img)
cv2.imshow("B", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()