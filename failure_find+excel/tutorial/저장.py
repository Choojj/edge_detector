import cv2
grayImg = cv2.imread('images/ironMan.PNG', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('images/grayIronMan.jpg', grayImg)
cv2.imshow('gray', grayImg)
cv2.waitKey(0)
cv2.destroyAllWindows()