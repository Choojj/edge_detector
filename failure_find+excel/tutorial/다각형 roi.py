import cv2
import numpy as np

path = "C:/Users/HSWB/Desktop/edge_detector/data1/original_01.png"
# original image
# -1 loads as-is so if it will be 3 or 4 channel as the original
image = cv2.imread(path, -1)
# mask defaulting to black for 3-channel and transparent for 4-channel
# (of course replace corners with yours)
mask = np.zeros(image.shape, dtype=np.uint8)

mask = cv2.bitwise_not(mask)

roi_corners = np.array([[[0, 0], [100, 0], [100, 100], [0, 100]]], dtype=np.int32)
# fill the ROI so it doesn't get wiped out when the mask is applied
channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (0,)*channel_count
print(ignore_mask_color)
cv2.fillPoly(mask, roi_corners, ignore_mask_color)
# from Masterfool: use cv2.fillConvexPoly if you know it's convex

# apply the mask
masked_image = cv2.bitwise_or(image, mask)

# save the result
cv2.imshow('image_masked.png', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()