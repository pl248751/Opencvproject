import numpy as np
import cv2 as cv
import argparse
import sys
import math
from collections import Counter


mark_radius = 1
mark_color = (0, 0, 255)

# parser = argparse.ArgumentParser(description='This sample demonstrates the camshift algorithm. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# parser.add_argument('video', type=str, help='big.mp4')
# args = parser.parse_args()
# args.image = 'big.mp4'
# cap = cv.VideoCapture(args.image)
cap = cv.VideoCapture('run.mp4')
VIDEO_W = cap.get(cv.CAP_PROP_FRAME_WIDTH)
VIDEO_H = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

# sys.exit(0)

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


out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (int(VIDEO_W), int(VIDEO_H)))


def connect_dots(src, dots):
    if not dots:
        return
    prev_pt = dots[0]
    for pt in dots:

        cv.line(src, prev_pt, pt, (0, 0, 255), 3, cv.LINE_AA)
        prev_pt = pt


def connect_lines(src, lines):
    prev_line = lines.pop()[:2]

    prev_pt = prev_line[:2]
    for line in lines:
        pt = line[:2]
        cv.line(src, prev_pt, pt, (0, 0, 255), 3, cv.LINE_AA)
        prev_pt = pt


def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    distance = int(distance)

    return distance


def slope(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return (y1 - y2) / (x1 - x2)


while True:
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply camshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        # Draw it on image
        # pts = cv.boxPoints(ret)
        # pts = np.int0(pts)

        # img2 = cv.polylines(frame, [pts], True, 255, 2)
        # img2 = cv.circle(frame, [pts], 63, (0, 0, 255), -1)
        img2 = cv.circle(frame,
                         (int(x) + int(VIDEO_W / 2), int(y) + 50),
                         5,
                         mark_color,
                         20,
                         cv.LINE_AA)
        x, y = ret[0]
        # img2 = cv.circle(frame,
        #                  (int(x), int(y - 20)),
        #                  mark_radius,
        #                  mark_color,
        #                  -1,
        #                  cv.LINE_AA)

        # image process
        dst = cv.Canny(img2, 50, 100, None, 3)

        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)

        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                # pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

                # if abs(slope(pt1, pt2)) > 0.3:
                x1, y1 = pt1
                x2, y2 = pt2
                # x1 = max(x1, 0)
                # x2 = max(x2, 0)
                # y1 = max(y1, 0)
                # y2 = max(y2, 0)
                # if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                cv.line(cdst, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)

        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
        # cv.imshow("Source", img2)

        # points = []
        ls = []
        rs = []
        total_slope = 0

        valid_lines = []

        if linesP is not None:

            # top

            centers = []

            l = linesP[0][0]
            pt1_prev = (l[0], l[1])
            pt2_prev = (l[2], l[3])
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                pt1 = (l[0], l[1])
                pt2 = (l[2], l[3])
                # points.append(pt1)
                # points.append(pt2)

                # point_slope = slope(pt1, pt2)
                # if 270 < l[1] < 544 and 270 < l[3] < 544 and (600 > l[0] > 400 or 240 < l[2] < 400):
                # if 1:
                if l[0] > 200 and l[0] < 750 or l[3] > 200 and l[3] < 750:
                    valid_lines.append(l)

                # ls.append(round(point_slope, 2))
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
                # distance_1 = dist(pt1, pt1_prev)
                # distance_2 = dist(pt1, pt2_prev)

                if abs([1] - l[3]) < 200 and abs(l[0] - l[2]) > 400:
                    pt3 = ((l[0] + l[2]) // 2, (l[3] + l[1]) // 2)
                    # print('\t', pt3)
                    centers.append(pt3)

                if abs(l[0] - l[2]) < 200 and abs([1] - l[3]) > 400:
                    pt4 = ((l[0] + l[2]) // 2, (l[3] + l[1]) // 2)
                    # print('\t', pt4)
                    centers.append(pt4)

                pt1_prev = pt1
                pt2_prev = pt2

            # top_x, top_y = sum(top_points[0]) / len(top_points), sum(top_points[1]) / len(top_points)
            # print(centers)
            # centers.sort()

            valid_lines.sort(key=lambda x: x[1])
            ls.append(valid_lines.pop())
            rs.append(valid_lines.pop())
            # for p in valid_lines:
            while valid_lines:
                valid_line = valid_lines.pop()
                point = valid_line[:2]
                if dist(point, ls[-1][:2]) < dist(point, rs[-1][:2]):
                    ls.append(valid_line)
                else:
                    rs.append(valid_line)
                # dist(p,)
                # print(p)
            # connect_lines(src, ls)
            # connect_lines(src, rs)
            real_centers = []
            for i in range(min(len(ls), len(rs))):
                # real_centers.append()

                lx1, ly1, lx2, ly2 = ls[i]
                rx1, ry1, rx2, ry2 = rs[i]
                if lx1 > 700 or lx2 > 700 or rx1 > 700 or rx2 > 700:
                    continue

                if abs(lx1 - rx1) > 200 and abs(ly1 - ry1) < 50:
                    center = ((lx1 + rx1) // 2, (ly1 + ry1) // 2)
                    real_centers.append(center)

                if abs(lx1 - rx1) < 200 and abs(ly1 - ry1) < 50:
                    center1 = ((lx1 + rx1) // 2, (ly1 + ry1) // 2)
                    real_centers.append(center1)

            # connect_dots(img2, real_centers)
            # connect_dots(img2, rs)
            total_x = sum([point[0] for point in real_centers])
            # total_y = sum([point[1] for point in real_centers])

            print(len(ls), len(rs), len(real_centers), mark_color)

            if real_centers and abs(total_x / len(real_centers) - VIDEO_W / 2) < 180:
                mark_color = (0, 255, 0)
            else:
                mark_color = (0, 0, 255)
            print(mark_color, len(real_centers), real_centers)
            # cv.waitKey()
            # return 0

        cv.imshow('img2', img2)
        out.write(img2)
        k = cv.waitKey(30) & 0xff
    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()
