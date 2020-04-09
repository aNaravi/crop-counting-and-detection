import cv2
import numpy as np
import argparse
import os
from itertools import product


def get_pixels(event, col, row, flags, param):
    global points_to_be_added, left_tops, right_bottoms

    if event == cv2.EVENT_LBUTTONDOWN:
        left_tops.append((row, col))

    if event == cv2.EVENT_LBUTTONUP:
        right_bottoms.append((row, col))

    if event == cv2.EVENT_RBUTTONDBLCLK:
        points_to_be_added.append((row, col))


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", default=".", required=False, help="path to images")
ap.add_argument("-n", "--inum", default=".", required=False, help="image number")
args = vars(ap.parse_args())

dir_images = args.get('dir') + '/images/'
dir_annotated = args.get('dir') + '/annotated/'
dir_binaries = args.get('dir') + '/binaries/'
inum = args.get('inum')

image = cv2.imread(dir_images + 'entry_' + inum + '.tif')
annotated = cv2.imread(dir_annotated + 'annotated_' + inum + '.tif')
binary = cv2.imread(dir_binaries + 'binary_' + inum + '.tif', cv2.IMREAD_GRAYSCALE)

if annotated is None:
    os.makedirs(dir_annotated, exist_ok=True)
    os.makedirs(dir_binaries, exist_ok=True)
    annotated = np.copy(image)
    binary = np.zeros(image.shape[:2])

points_to_be_added, left_tops, right_bottoms = [], [], []
while True:
    for p in points_to_be_added:
        binary[p[0],p[1]] = 255
        cv2.circle(annotated, (p[1], p[0]), 3, (0,0,255), -1)
    points_to_be_added = []

    for i in range(0, len(left_tops)):
        corner1, corner2 = left_tops[i], right_bottoms[i]
        rows = list(range(corner1[0], corner2[0] + 1))
        cols = list(range(corner1[1], corner2[1] + 1))
        pixels = list(product(rows, cols))

        for pixel in product(rows, cols):
            color = image[pixel[0], pixel[1], :]
            annotated[pixel[0], pixel[1], :] = color
            binary[pixel[0], pixel[1]] = 0
    left_tops, right_bottoms = [], []

    cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("win", 2000, 1000)
    cv2.setMouseCallback("win", get_pixels)
    cv2.imshow("win", annotated)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()

cv2.imwrite(dir_binaries + 'binary_' + inum + ".tif", binary)
cv2.imwrite(dir_annotated + 'annotated_' + inum + ".tif", annotated)
