import cv2
import numpy as np
from itertools import product


def get_pixels(event, col, row, flags, param):
    global points_to_be_added, left_tops, right_bottoms

    if event == cv2.EVENT_LBUTTONDOWN:
        left_tops.append((row, col))

    if event == cv2.EVENT_LBUTTONUP:
        right_bottoms.append((row, col))

    if event == cv2.EVENT_RBUTTONDBLCLK:
        points_to_be_added.append((row, col))


points_to_be_added, left_tops, right_bottoms = [], [], []
dir_images = 'Images/Week_3_-_29th_Jan_19/Window1/CNN_training/'
name = '113'
decorr = cv2.imread(dir_images + name + '_decorred.tif')
annotated = cv2.imread(dir_images + name + '_annotated.tif')
binary = cv2.imread(dir_images + name + '_binary.tif')

while True:
    for p in points_to_be_added:
        binary[p[0],p[1],:] = np.array([255,255,255])
        # cv2.circle(binary, (p[1], p[0]), 3, (255,255,255), -1)
        cv2.circle(annotated, (p[1], p[0]), 3, (0,0,255), -1)
    points_to_be_added = []

    for i in range(0, len(left_tops)):
        corner1, corner2 = left_tops[i], right_bottoms[i]
        rows = list(range(corner1[0], corner2[0] + 1))
        cols = list(range(corner1[1], corner2[1] + 1))
        pixels = list(product(rows, cols))

        for pixel in product(rows, cols):
            color = decorr[pixel[0], pixel[1], :]
            annotated[pixel[0], pixel[1], :] = color
            binary[pixel[0], pixel[1], :] = np.array([0, 0, 0])
    left_tops, right_bottoms = [], []

    cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("win", 2000, 1000)
    cv2.setMouseCallback("win", get_pixels)
    cv2.imshow("win", annotated)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()


cv2.imwrite(dir_images + name + "_binary.tif", binary)
cv2.imwrite(dir_images + name + "_annotated.tif", annotated)
