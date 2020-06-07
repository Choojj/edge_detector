import cv2
import numpy as np

def extraction(image): # 원본에서 roi 추출
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # cv2.imshow("1", blurred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # cv2.imshow("1", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edge = cv2.Canny(gray, 50, 100)
    # cv2.imshow("1", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edge = cv2.bitwise_not(edge)
    # cv2.imshow("1", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, contours[0], -1, (0, 0, 0), 1, cv2.LINE_8)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in enumerate(contours): # 차영상에서 도형 정보 수집
        epsilon = 0.005 * cv2.arcLength(cnt[1], True)
        approx = cv2.approxPolyDP(cnt[1], epsilon, True)

        if (80000 < cv2.contourArea(cnt[1]) < 1000000 and len(approx) >= 8):
            # 넓이 한번 거른후, 엣지마다 바깥, 안 컨투어가 생기는데 안쪽 컨투어 사용, 꼭지점 8개 이상
            # 컨투어 내부에 컨투어가 없다는걸 전제로 했는데 구멍이 있을경우 X (회전보정때 사용해서 변경도 X)
            # 엣지에 컨투어 2개가 생기는경우 큰쪽이 작은쪽의 부모인가(같거나 더 작은가)로 확인해야할듯 -> 모든경우 일치
            # 일단 귀찮아서 마지막걸로 바꿨는데 나름 잘되는듯
            for i in range(len(approx)):
                print(approx[i][0], end = " ")
            print()
            left, top, width, height = cv2.boundingRect(cnt[1])
            dst = image.copy()
            dst = image[top - 10:top + height + 10, left - 10:left + width + 10]
    return dst

def base_rectangle_point(list):
    new_list = []

    Max = Max2 = -float("inf")
    Min = Min2 = float("inf")
    Max_num = Max2_num = 0
    Min_num = Min2_num = 0

    for i in range(len(list)):
        if (list[i][0][0] >= Max):
            Max2 = Max
            Max2_num = Max_num
            Max = list[i][0][0]
            Max_num = i
        elif (Max2 < list[i][0][0] < Max):
            Max2 = list[i][0][0]
            Max2_num = i

        if (list[i][0][0] <= Min):
            Min2 = Min
            Min2_num = Min_num
            Min = list[i][0][0]
            Min_num = i
        elif (Min2 > list[i][0][0] > Min):
            Min2 = list[i][0][0]
            Min2_num = i

    if (list[Min_num][0][1] > list[Min2_num][0][1]):
        new_list.append([])
        new_list[0].append(list[Min2_num][0][0])
        new_list[0].append(list[Min2_num][0][1])
        new_list.append([])
        new_list[1].append(list[Min_num][0][0])
        new_list[1].append(list[Min_num][0][1])
        if (list[Max_num][0][1] > list[Max2_num][0][1]):
            new_list.append([])
            new_list[2].append(list[Max2_num][0][0])
            new_list[2].append(list[Max2_num][0][1])
            new_list.append([])
            new_list[3].append(list[Max_num][0][0])
            new_list[3].append(list[Max_num][0][1])
        else:
            new_list.append([])
            new_list[2].append(list[Max_num][0][0])
            new_list[2].append(list[Max_num][0][1])
            new_list.append([])
            new_list[3].append(list[Max2_num][0][0])
            new_list[3].append(list[Max2_num][0][1])
    else:
        new_list.append([])
        new_list[0].append(list[Min_num][0][0])
        new_list[0].append(list[Min_num][0][1])
        new_list.append([])
        new_list[1].append(list[Min2_num][0][0])
        new_list[1].append(list[Min2_num][0][1])
        if (list[Max_num][0][1] > list[Max2_num][0][1]):
            new_list.append([])
            new_list[2].append(list[Max2_num][0][0])
            new_list[2].append(list[Max2_num][0][1])
            new_list.append([])
            new_list[3].append(list[Max_num][0][0])
            new_list[3].append(list[Max_num][0][1])
        else:
            new_list.append([])
            new_list[2].append(list[Max_num][0][0])
            new_list[2].append(list[Max_num][0][1])
            new_list.append([])
            new_list[3].append(list[Max2_num][0][0])
            new_list[3].append(list[Max2_num][0][1])

    return new_list

# [[9.0, 268.0], [9.0, 396.0], [656.0, 268.0], [656.0, 396.0]]

path = "C:/Users/HSWB/Desktop/edge_detector/data1/modified_03.png"

image = cv2.imread(path, cv2.IMREAD_COLOR)
image = image[225:919, 82:1816]

roi = extraction(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 1000, 0.001, 5, blockSize=3, useHarrisDetector=True, k=0.05)

AA = np.array([1918, 444], np.float64)

for i in range(len(corners)):
    print(corners[i][0]>[1000,500], end = " ")
    cv2.circle(image, (int(corners[i][0][0]), int(corners[i][0][1])), 5, (0, 0, 255), 2)
print()

# print(corners > [1000, 500])

anchor_point = base_rectangle_point(corners)

cv2.imshow("A", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 오차 1픽셀 안쪽