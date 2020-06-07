import numpy as np
import cv2
import pytesseract

image = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/ocr/data/ALPHABET.png")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

invert = cv2.bitwise_not(blurred)
cv2.imshow("invert", invert)
cv2.waitKey(0)
cv2.destroyAllWindows()

edge = cv2.Canny(invert, 50, 200, apertureSize = 3)
cv2.imshow("edge", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

rho = 1 # r의 범위 (0 ~ 1)
theta = np.pi / 180 # 세타의 범위 (0 ~ 180)
threshold = 100 # 만나는 점의 기준 / 정확도
minLineLength = 100 # 선의 최소 길이
maxLineGap = 0 # 선 간격
lines = cv2.HoughLinesP(edge, rho, theta, threshold, minLineLength, maxLineGap)
for i in range(len(lines)):
    cv2.line(edge, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 10)
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
    top, left, height, width = cv2.boundingRect(cnt)
    dst = image.copy()
    dst = dst[left:left + width, top:top + height]
    cv2.imshow("result", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

txt = pytesseract.image_to_string(dst) #config = '--psm 13')
print(txt)

gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
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

name = 0
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    name += 1
    alphabet = dst.copy() 
    alphabet = dst[y:y + h, x:x + w] # y:dy, x:dx
    cv2.imwrite("C:/Users/HSWB/Desktop/edge_detector/ocr/data/ocr_" + str(name) +".png", alphabet)

    cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()