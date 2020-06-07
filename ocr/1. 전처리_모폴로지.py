import cv2
import numpy as np

image = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/ocr/data/modified_01.png")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

erosion = cv2.erode(gray, kernel, iterations = 1)
cv2.imshow("erosion", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

dilation = cv2.dilate(gray, kernel, iterations = 1)
cv2.imshow("dilation", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,kernel)
cv2.imshow("closing", closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("gradient", gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("tophat", tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("blackhat", blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()