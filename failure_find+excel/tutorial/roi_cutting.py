import cv2

src = cv2.imread("C:/Users/HSWB/Desktop/edge_detector/sample/original_01.png", cv2.IMREAD_COLOR)

dst = src.copy() 
dst = src[10:60, 200:700] # y:dy, x:dx

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()