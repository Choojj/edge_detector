from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

imageA = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/sample/original_01.png")
imageB = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/sample/modified_13.png")

height1, width1, channel1 = imageA.shape
height2, width2, channel2 = imageB.shape

if height1 != height2 or width1 != width2:
    imageB = cv2.resize(imageB, (width1, height1), interpolation = cv2.INTER_LINEAR)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if w > 10 or h > 10:
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
#cv2.imshow("Diff", diff)
#cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
두개 이미지 비교 후 다른 부분 표시
이미지 크기가 서로 같아야함 -> 사진 비율을 같게 맞춰줘야 함
"""