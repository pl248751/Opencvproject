import cv2
import numpy as np
import matplotlib.pyplot as plt

# To open matplotlib in interactive mode
# %matplotlib qt

# Load the image
img = cv2.imread('circle.jpeg')

# Create a copy of the image
img_copy = np.copy(img)

# Convert to RGB so as to display via matplotlib
# Using Matplotlib we can easily find the coordinates
# of the 4 points that is essential for finding the
# transformation matrix
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

plt.imshow(img_copy)

# to calculate the transformation matrix

# def find_points(pt_A, pt_B, pt_C, pt_D):


def find_points(input_pts):
    # input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    pt_A, pt_B, pt_C, pt_D = input_pts[0], input_pts[1], input_pts[2], input_pts[3]
    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])
    return output_pts, maxWidth, maxHeight


A = 1690, 2009
B = 1855, 87
C = 794, 234
D = 241, 909
input_pts = np.float32([C, D, A, B])

# A = 794, 254
# B = 241, 909
# C = 2090, 2009
# D = 1855, 87
# input_pts = np.float32([A, B, C, D])

# input_pts = np.float32([[80, 1286], [3890, 1253], [3890, 122], [450, 115]])
# output_pts = np.float32([[100, 100], [100, 3900], [2200, 3900], [2200, 100]])
output_pts, maxWidth, maxHeight = find_points(input_pts)

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)

# Apply the perspective transformation to the image
out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

# Display the transformed image
plt.imshow(out)
cv2.imwrite('circle_regular.png', out)
plt.show()
