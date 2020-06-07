import cv2
import numpy as np

blank_image = np.zeros((500, 500, 4), np.uint8)
blank_image = cv2.bitwise_not(blank_image)

red_color = (0, 0, 255, 255)

cv2.circle(blank_image, (255, 255), 100, red_color, 5)


cv2.imshow("iamge", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()