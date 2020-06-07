import cv2
import numpy as np

def retouching(path):
    rotation_point = []

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    # cv2.imshow("result", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("result", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edge = cv2.Canny(gray, 50, 100)
    # cv2.imshow("result", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edge = cv2.bitwise_not(edge)
    # cv2.imshow("result", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, contours[0], -1, (0, 0, 0), 1, cv2.LINE_8)
    # cv2.imshow("result", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
                    rotation_point.append([])
                    rotation_point[0].append(approx[Min2_num][0][0])
                    rotation_point[0].append(approx[Min2_num][0][1])
                    rotation_point.append([])
                    rotation_point[1].append(approx[Min_num][0][0])
                    rotation_point[1].append(approx[Min_num][0][1])
                    if (approx[Max_num][0][1] > approx[Max2_num][0][1]):
                        rotation_point.append([])
                        rotation_point[2].append(approx[Max2_num][0][0])
                        rotation_point[2].append(approx[Max2_num][0][1])
                        rotation_point.append([])
                        rotation_point[3].append(approx[Max_num][0][0])
                        rotation_point[3].append(approx[Max_num][0][1])
                    else:
                        rotation_point.append([])
                        rotation_point[2].append(approx[Max_num][0][0])
                        rotation_point[2].append(approx[Max_num][0][1])
                        rotation_point.append([])
                        rotation_point[3].append(approx[Max2_num][0][0])
                        rotation_point[3].append(approx[Max2_num][0][1])
                else:
                    rotation_point.append([])
                    rotation_point[0].append(approx[Min_num][0][0])
                    rotation_point[0].append(approx[Min_num][0][1])
                    rotation_point.append([])
                    rotation_point[1].append(approx[Min2_num][0][0])
                    rotation_point[1].append(approx[Min2_num][0][1])
                    if (approx[Max_num][0][1] > approx[Max2_num][0][1]):
                        rotation_point.append([])
                        rotation_point[2].append(approx[Max2_num][0][0])
                        rotation_point[2].append(approx[Max2_num][0][1])
                        rotation_point.append([])
                        rotation_point[3].append(approx[Max_num][0][0])
                        rotation_point[3].append(approx[Max_num][0][1])
                    else:
                        rotation_point.append([])
                        rotation_point[2].append(approx[Max_num][0][0])
                        rotation_point[2].append(approx[Max_num][0][1])
                        rotation_point.append([])
                        rotation_point[3].append(approx[Max2_num][0][0])
                        rotation_point[3].append(approx[Max2_num][0][1])
                print(rotation_point)

    h, w = image.shape[:2]

    pts1 = np.float32(rotation_point)
    pts2 = np.float32([[637, 615], [637, 741], [1282, 615], [1282, 741]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    return image, cv2.warpPerspective(image, M, (w, h))

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
            print(len(approx), cnt[0])
            for i in range(len(approx)):
                print(i, approx[i][0], end = " ")
            print(hierarchy[0][cnt[0]])
            left, top, width, height = cv2.boundingRect(cnt[1])
            dst = image.copy()
            dst = image[top - 0:top + height + 0, left - 0:left + width + 0]
    return dst

def absdiff(image1, image2):
    height, width = 600, 1000
    blank_image = np.zeros((height, width, 3), np.uint8)
    blank_image1 = cv2.bitwise_not(blank_image)
    blank_image2 = cv2.bitwise_not(blank_image)

    result1_y, result1_x = roi1.shape[:2]
    result2_y, result2_x = roi2.shape[:2]
    blank_image1[height - result1_y:height, 0:result1_x] = image1
    blank_image2[height - result2_y:height, 0:result2_x] = image2

    gray1 = cv2.cvtColor(blank_image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(blank_image2, cv2.COLOR_BGR2GRAY)

    ret, binary1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, binary2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    absdiff = cv2.absdiff(binary1, binary2)

    return blank_image1, blank_image2, absdiff

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ


path1 = "C:/Users/HSWB/Desktop/edge_detector/data1/original_01.png"
path2 = "C:/Users/HSWB/Desktop/edge_detector/data1/rotation_10_02.png"

clip1, retouch_clip1 = retouching(path1)
clip2, retouch_clip2 = retouching(path2)

roi1 = extraction(retouch_clip1)
roi2 = extraction(retouch_clip2)

roi1, roi2, absdiff = absdiff(roi1, roi2)

cv2.imshow("result1", roi1)
cv2.imshow("result2", roi2)
cv2.imshow("result3", absdiff)
cv2.waitKey(0)
cv2.destroyAllWindows()
