import matplotlib.pyplot as plt
import numpy as np
import cv2

# To open matplotlib in interactive mode
# %matplotlib qt

# Load the image
src = cv2.imread('circle_regular.png')
blur_img = cv2.medianBlur(src, 13)
gray_scale = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

# Create a copy of the image
img_copy = np.copy(src)
circles = cv2.HoughCircles(gray_scale, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=80, param2=30, minRadius=575, maxRadius=640)

circles = np.uint16(np.round(circles))

for i in circles[0, :1]:
    cv2.circle(src, (i[0], i[1]), i[2], (0, 0, 255), 5)
    cv2.circle(src, (i[0], i[1]), 2, (0, 0, 0), 5)

cv2.imwrite('circle_regular_marked.png', src)
cv2.imshow('Results', src)
cv2.waitKey(0)
cv2.destroyAllWindows()
# img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

# plt.imshow(img_copy)

# # to calculate the transformation matrix
# input_pts = np.float32([[80, 1286], [3890, 1253], [3890, 122], [450, 115]])
# output_pts = np.float32([[100, 100], [100, 3900], [2200, 3900], [2200, 100]])

# # Display the transformed image
# plt.imshow(out)
# plt.show()

