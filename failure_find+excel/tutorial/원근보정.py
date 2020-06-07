import numpy as np
import cv2

path = path = "C:/Users/HSWB/Desktop/edge_detector/data1/rotation_01.png"

img = cv2.imread(path)
h, w = img.shape[:2]

pts1 = np.float32([[779, 616], [778, 742], [1234, 615], [1234, 742]])
pts2 = np.float32([[637, 615], [637, 741], [1282, 615], [1282, 741]])

M = cv2.getPerspectiveTransform(pts1, pts2)

img2 = cv2.warpPerspective(img, M, (w, h))

cv2.imshow("A", img)
cv2.imshow("B", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()