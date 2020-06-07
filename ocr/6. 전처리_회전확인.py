import numpy as np
import cv2
import pytesseract

def text_extraction(path):
    image = cv2.imread(path)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # cv2.imshow("blurred", blurred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    invert = cv2.bitwise_not(blurred)
    # cv2.imshow("invert", invert)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edge = cv2.Canny(invert, 50, 200, apertureSize = 3)
    # cv2.imshow("edge", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rho = 1 # r의 범위 (0 ~ 1)
    theta = np.pi / 180 # 세타의 범위 (0 ~ 180)
    threshold = 100 # 만나는 점의 기준 / 정확도
    minLineLength = 100 # 선의 최소 길이
    maxLineGap = 0 # 선 간격
    lines = cv2.HoughLinesP(edge, rho, theta, threshold, minLineLength, maxLineGap)
    for i in range(len(lines)):
        cv2.line(edge, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 10)
    # cv2.imshow('line remove', edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    dilation = cv2.dilate(edge, kernel, iterations = 10)
    # cv2.imshow("dilation", dilation)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        top, left, height, width = cv2.boundingRect(cnt)
        dst = image.copy()
        dst = dst[left:left + width, top:top + height]
        # cv2.imshow("result", dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    return dst


path = "C:/Users/HSWB/Desktop/edge_detector/ocr/data/rotation_165.png"

image = text_extraction(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

invert = cv2.bitwise_not(gray)
# cv2.imshow("invert", invert)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
	angle = -(90 + angle)

else:
	angle = -angle

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated1 = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
M = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated2 = cv2.warpAffine(rotated1, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
rotated3 = cv2.warpAffine(rotated2, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
rotated4 = cv2.warpAffine(rotated3, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)

cv2.imshow("Input", image)
cv2.imshow("Rotated1", rotated1)
cv2.imshow("Rotated2", rotated2)
cv2.imshow("Rotated3", rotated3)
cv2.imshow("Rotated4", rotated4)
cv2.waitKey(0)
cv2.destroyAllWindows()

rotated = [rotated1, rotated2, rotated3, rotated4]

break_trig2 = False
for i in range(4):
    if (break_trig2 == True):
        break

    image = rotated[i]
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    invert = cv2.bitwise_not(gray)
    cv2.imshow("invert", invert)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    alphabet = image.copy() 
    alphabet = image[y:y + h, x:x + w]
    cv2.imshow("alphabet", alphabet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blank_image = np.zeros((35, 35, 3), np.uint8)
    blank_imageA = cv2.bitwise_not(blank_image)
    grayA = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY)
    blank_imageA = cv2.cvtColor(blank_imageA, cv2.COLOR_BGR2GRAY)
    ret, resultA = cv2.threshold(grayA, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resultA_y, resultA_x = resultA.shape
    blank_imageA[35 - resultA_y:35, 0:resultA_x] = resultA

    break_trig = False
    for j in range(1, 37):
        if (break_trig == True):
            break

        ocr_XX = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/ocr/data/ocr_" + str(j) + ".png")

        blank_imageB = cv2.bitwise_not(blank_image)
        grayB = cv2.cvtColor(ocr_XX, cv2.COLOR_BGR2GRAY)
        blank_imageB = cv2.cvtColor(blank_imageB, cv2.COLOR_BGR2GRAY)
        ret, resultB = cv2.threshold(grayB, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resultB_y, resultB_x = resultB.shape
        blank_imageB[35 - resultB_y:35, 0:resultB_x] = resultB

        absdiff = cv2.absdiff(blank_imageA, blank_imageB)

        contours, hierarchy = cv2.findContours(absdiff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(j)

        
        same = True
        for cnt in enumerate(contours):
            area = cv2.contourArea(cnt[1])
            print(cnt[0], area)
            if (area > 5):
                same = False
                break

        if (same):
            break_trig = True
            break_trig2 = True

            right_text_image = rotated[i]
            print("SAME!!!!")
            
        cv2.imshow("blank_imageA", blank_imageA)
        cv2.imshow("blank_imageB", blank_imageB)
        cv2.imshow("absdiff", absdiff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

cv2.imshow("right_text_image", right_text_image)
cv2.waitKey(0)
cv2.destroyAllWindows()