import numpy as np
import cv2
import pytesseract

image = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/ocr/data/rotation_0.png")
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

image = dst

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

invert = cv2.bitwise_not(gray)
cv2.imshow("invert", invert)
cv2.waitKey(0)
cv2.destroyAllWindows()

thresh = cv2.threshold(invert, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
print(cv2.minAreaRect(coords))

if angle < -45:
	angle = -(90 + angle)

else:
	angle = -angle

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", image)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)

txt = pytesseract.image_to_string(rotated) #config = '--psm 13')
print(txt)

"""
OCR Engine modes(–oem):
0 - Legacy engine only.
1 - Neural nets LSTM engine only.
2 - Legacy + LSTM engines.
3 - Default, based on what is available.
Page segmentation modes(–psm):
0 - Orientation and script detection (OSD) only.
1 - Automatic page segmentation with OSD.
2 - Automatic page segmentation, but no OSD, or OCR.
3 - Fully automatic page segmentation, but no OSD. (Default)
4 - Assume a single column of text of variable sizes.
5 - Assume a single uniform block of vertically aligned text.
6 - Assume a single uniform block of text.
7 - Treat the image as a single text line.
8 - Treat the image as a single word.
9 - Treat the image as a single word in a circle.
10 - Treat the image as a single character.
11 - Sparse text. Find as much text as possible in no particular order.
12 - Sparse text with OSD.
13 - Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
"""