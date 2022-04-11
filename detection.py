import sys
import math
import cv2 as cv
import numpy as np
from collections import Counter


def connect_dots(src, dots):
    prev_pt = dots[0]
    for pt in dots:
        cv.line(src, prev_pt, pt, (0, 255), 3, cv.LINE_AA)
        prev_pt = pt


def connect_lines(src, lines):
    prev_line = lines.pop()[:2]

    prev_pt = prev_line[:2]
    for line in lines:
        pt = line[:2]
        cv.line(src, prev_pt, pt, (0, 0, 255), 3, cv.LINE_AA)
        prev_pt = line[-2:]


def show_lines(src, lines):
    for line in lines:
        pt1 = line[:2]
        pt2 = line[-2:]
        cv.line(src, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
        prev_pt = line[-2:]


def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    distance = int(distance)

    return distance


def main(argv, default_file=''):
    # img = cv.imread('right.jpg')
    filename = argv[0] if len(argv) > 0 else default_file
    img = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Second, process edge detection use Canny.

    low_threshold = 380
    high_threshold = 5
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)
    # Then, use HoughLinesP to get the lines. You can adjust the parameters for better performance.

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)

    valid_lines = []

    for line in lines:
        print(line)
        # for x1, y1, x2, y2 in line:

        x1, y1, x2, y2 = line[0]
        cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)
        cv.imshow("org", img)
        # if y1 <= 600:
        valid_lines.append(line[0])

    valid_lines.sort(key=lambda x: x[1])
    valid_lines.pop(0)
    valid_lines.pop()
    print('------sorted-----')
    print(valid_lines)
    print('------sorted-----')

    ls = []
    rs = []

    # valid_lines.pop()
    # valid_lines.pop()

    ls.append(valid_lines.pop(0))
    # rs.append(valid_lines.pop(0))
    print("------before---")
    print(ls)
    print(rs)
    print("------before---")

    count = 0
    for line in valid_lines:
        count += 1
        point = line[:2]

        # if not rs and dist(point, ls[-1][-2:]) < 330:
        # if dist(point, ls[-1][-2:]) < dist(point, rs[-1][-2:]):
        ls_dist = dist(point, ls[-1][-2:])

        rs_dist = ls_dist / 2
        if rs:
            rs_dist = dist(point, rs[-1][-2:])

        # dist(point, rs[-1][-2:]):
        if ls_dist < rs_dist:
            ls.append(line)
        else:
            rs.append(line)
        # if count == 10:
        #     break
    # Finally, draw the lines on your srcImage.
    print("------after---")
    print(ls)
    print(rs)
    print("------after---")

    rs.sort(key=lambda x: x[1])
    ls.sort(key=lambda x: x[1])

    # show_lines(img, valid_lines[:15])
    show_lines(img, rs)
    # Draw the lines on the  image

    real_centers = []
    for i in range(min(len(ls), len(rs))):
        # real_centers.append()

        lx1, ly1, lx2, ly2 = ls[i]
        rx1, ry1, rx2, ry2 = rs[i]

        # if abs(lx1 - rx1) > 50 and abs(ly1 - ry1) < 50:
        center = ((lx1 + rx1) // 2, (ly1 + ry1) // 2)
        # if center - lx1
        real_centers.append(center)

        # if abs(lx1 - rx1) < 50 and abs(ly1 - ry1) > 50:
        #     center1 = ((lx1 + rx1) // 2, (ly1 + ry1) // 2)
        #     real_centers.append(center1)

    lx1, ly1, lx2, ly2 = ls[-1]
    rx1, ry1, rx2, ry2 = rs[-1]
    center = ((lx1 + rx1) // 2, (ly1 + ry1) // 2)
    real_centers.append(center)
    print('------centers------')
    print(real_centers)
    connect_dots(img, real_centers)

    lines_edges = cv.addWeighted(img, 0.8, line_image, 1, 0)

    cv.imshow("Source", line_image)
    cv.imshow("org", img)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    default_file = 'l.jpg'
    main(sys.argv[1:], default_file)
    default_file = 'r.jpg'
    main(sys.argv[1:], default_file)
