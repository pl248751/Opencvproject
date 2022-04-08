"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
from collections import Counter


def connect_dots(src, dots):
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
    cv.imshow("Source", src)

    points = []
    ls = []
    rs = []
    total_slope = 0

    valid_lines = []

    if linesP is not None:

        # top
        top_min_y = 2000
        top_min_x = 0

        centers = []

        l = linesP[0][0]
        pt1_prev = (l[0], l[1])
        pt2_prev = (l[2], l[3])
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            pt1 = (l[0], l[1])
            pt2 = (l[2], l[3])
            points.append(pt1)
            points.append(pt2)

            # point_slope = slope(pt1, pt2)
            if 100 < l[1] < 930 and 100 < l[3] < 730 and (950 > l[0] > 350 or 350 < l[2] < 1300):
                # if 1:
                print(l)
                valid_lines.append(l)

                # ls.append(round(point_slope, 2))
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
                # distance_1 = dist(pt1, pt1_prev)
                # distance_2 = dist(pt1, pt2_prev)
                # print(distance_2, distance_1)

                if abs([1] - l[3]) < 200 and abs(l[0] - l[2]) > 400:
                    pt3 = ((l[0] + l[2]) // 2, (l[3] + l[1]) // 2)
                    print('\t', pt3)
                    centers.append(pt3)

                if abs(l[0] - l[2]) < 200 and abs([1] - l[3]) > 400:
                    pt4 = ((l[0] + l[2]) // 2, (l[3] + l[1]) // 2)
                    print('\t', pt4)
                    centers.append(pt4)

            pt1_prev = pt1
            pt2_prev = pt2

        # top_x, top_y = sum(top_points[0]) / len(top_points), sum(top_points[1]) / len(top_points)
        print('-------')
        print(centers)
        centers.sort()

        print('----all lines---')
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

            if abs(lx1 - rx1) > 50 and abs(ly1 - ry1) < 50:
                center = ((lx1 + rx1) // 2, (ly1 + ry1) // 2)
                real_centers.append(center)

            if abs(lx1 - rx1) < 50 and abs(ly1 - ry1) > 50:
                center1 = ((lx1 + rx1) // 2, (ly1 + ry1) // 2)
                real_centers.append(center1)

        connect_dots(src, real_centers)

        # dp1 = [844, 673, 878, 600]
        # dp2 = [849, 660, 872, 610]
        # dp1 = valid_lines[-1]
        # dp2 = valid_lines[-2]

        # cv.line(src, dp1[:2], dp1[-2:], (255, 0, 0), 3, cv.LINE_AA)
        # cv.line(src, dp2[:2], dp2[-2:], (255, 0, 0), 3, cv.LINE_AA)

        '''
        prev_pt = centers[0]
        for pt in centers[1:]:
            cv.line(src, prev_pt, pt, (0, 0, 255), 3, cv.LINE_AA)
            prev_pt = pt

        '''
    #             x1, y1 = pt1
    #             x2, y2 = pt2
    #             all_xs.append(x1)
    #             all_xs.append(x2)

    #             all_ys.append(y1)
    #             all_ys.append(y2)

    #             # base
    #             # if l[0] < base_min_x:
    #             #     base_min_x = l[0]
    #             #     base_min_y = l[1]
    #             # if l[2] < base_min_x:
    #             #     base_min_x = l[2]
    #             #     base_min_y = l[3]

    #             # if l[1] > base_max_y:
    #             #     base_max_y = l[1]
    #             #     base_max_x = l[2]
    #             # if l[3] > base_max_y:
    #             #     base_max_y = l[3]
    #             #     base_max_x = l[2]

    #             # top
    #             # if len(top_points) < 4:
    #             #     if l[1] < top_points[-1][1]:
    #             #         top_min_y = l[1]
    #             #         top_points.append((l[0], top_min_y))
    #             #     if l[2] < top_points[-1][1]:
    #             #         top_min_y = l[3]
    #             #         top_points.append((l[2], top_min_y))
    #             points.sort(key=lambda x: x[1])

    #     top_points = points[:2]
    #     # print(top_points)
    #     # ls.sort()
    #     print(sum(ls) / len(ls))
    #     print(ls)
    #     stat = Counter(ls)
    #     s_left, s_right = stat.most_common(2)
    #     print(s_left[0], s_right[0])

        # center_x, center_y = sum(all_xs) / len(all_xs), sum(all_ys) / len(all_ys)
        # top_x, top_y = sum(top_points[0]) / len(top_points), sum(top_points[1]) / len(top_points)
        # cv.line(src, (int(top_x), int(top_y)), (int(center_x), int(center_y)), (0, 255, 0), 5)

    #     # y = kx + b
    #     k = (s_left[0] + s_right[0]) / 2q
    #     b = center_y - center_x * k

    #     top_point = (int((100 - b) / k), 100)

    #     if s_left[0] * s_right[0] < 0:
    #         top_point = (int(center_x), 0)

    #     cv.arrowedLine(src, (int(center_x), int(center_y)), top_point, (0, 255, 0), 5)
    #     cv.imwrite(filename + 'maked.png', src)
    #     print(center_x, center_y)

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
