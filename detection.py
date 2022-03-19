"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
from collections import Counter


def slope(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return (y1 - y2) / (x1 - x2)


all_xs = []
all_ys = []


def main(argv, default_file=''):

    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    dst = cv.Canny(src, 50, 200, None, 3)

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
            # print(pt1, pt2)
            # if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            cv.line(cdst, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    points = []
    ls = []
    total_slope = 0

    if linesP is not None:

        # top
        top_min_y = 2000
        top_min_x = 0

        for i in range(0, len(linesP)):
            l = linesP[i][0]
            pt1 = (l[0], l[1])
            pt2 = (l[2], l[3])
            points.append(pt1)
            points.append(pt2)

            distance = math.dist(pt1, pt2)

            point_slope = slope(pt1, pt2)
            if abs(point_slope) > 0.3 and 250 > distance:

                ls.append(round(point_slope, 2))
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

                x1, y1 = pt1
                x2, y2 = pt2
                all_xs.append(x1)
                all_xs.append(x2)

                all_ys.append(y1)
                all_ys.append(y2)

                # base
                # if l[0] < base_min_x:
                #     base_min_x = l[0]
                #     base_min_y = l[1]
                # if l[2] < base_min_x:
                #     base_min_x = l[2]
                #     base_min_y = l[3]

                # if l[1] > base_max_y:
                #     base_max_y = l[1]
                #     base_max_x = l[2]
                # if l[3] > base_max_y:
                #     base_max_y = l[3]
                #     base_max_x = l[2]

                # top
                # if len(top_points) < 4:
                #     if l[1] < top_points[-1][1]:
                #         top_min_y = l[1]
                #         top_points.append((l[0], top_min_y))
                #     if l[2] < top_points[-1][1]:
                #         top_min_y = l[3]
                #         top_points.append((l[2], top_min_y))
                points.sort(key=lambda x: x[1])

        top_points = points[:2]
        # print(top_points)
        # ls.sort()
        print(sum(ls) / len(ls))
        print(ls)
        stat = Counter(ls)
        s_left, s_right = stat.most_common(2)
        print(s_left[0], s_right[0])

        center_x, center_y = sum(all_xs) / len(all_xs), sum(all_ys) / len(all_ys)
        top_x, top_y = sum(top_points[0]) / len(top_points), sum(top_points[1]) / len(top_points)
        # cv.line(src, (int(top_x), int(top_y)), (int(center_x), int(center_y)), (0, 255, 0), 5)

        # y = kx + b
        k = (s_left[0] + s_right[0]) / 2
        b = center_y - center_x * k

        top_point = (int((100 - b) / k), 100)

        if s_left[0] * s_right[0] < 0:
            top_point = (int(center_x), 0)

        cv.arrowedLine(src, (int(center_x), int(center_y)), top_point, (0, 255, 0), 5)
        print(center_x, center_y)

    cv.imshow("Source", src)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    default_file = 'flat.jpeg'
    main(sys.argv[1:], default_file)
    default_file = 'tilt_r.jpeg'
    main(sys.argv[1:], default_file)
    default_file = 'tilt_l.jpeg'
    main(sys.argv[1:], default_file)
