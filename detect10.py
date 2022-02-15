import numpy as np
import cv2 as cv
import argparse

mark_radius = 10
mark_color = (255, 0, 0)

# parser = argparse.ArgumentParser(description='This sample demonstrates the camshift algorithm. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# parser.add_argument('video', type=str, help='big.mp4')
# args = parser.parse_args()
# args.image = 'big.mp4'
# cap = cv.VideoCapture(args.image)
cap = cv.VideoCapture('big.mp4')
# take first frame of the video
ret, frame = cap.read()
# setup initial location of window
x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y + h, x:x + w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

# out = cv.VideoWriter('output.mp4', -1, 20.0, (1280, 720))


out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))

while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply camshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)

        # img2 = cv.polylines(frame, [pts], True, 255, 2)
        # img2 = cv.circle(frame, [pts], 63, (0, 0, 255), -1)
        img2 = cv.circle(frame,
                         (int(x), int(y - 20)),
                         275,
                         mark_color,
                         4,
                         cv.LINE_AA)
        x, y = ret[0]
        img2 = cv.circle(frame,
                         (int(x), int(y - 20)),
                         mark_radius,
                         mark_color,
                         -1,
                         cv.LINE_AA)
        # print(ret)
        print(hsv)

        cv.imshow('img2', img2)
        out.write(img2)
        k = cv.waitKey(30) & 0xff
        if k == 5:
            break
    else:
        break

cap.release()
out.release()
# cv.destroyAllWindows()
