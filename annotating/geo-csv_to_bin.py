import os
import numpy as np
import cv2
import rasterio
import csv
import math

img_dir = '/home/Image_Processing/July/images_w7/'
bin_dir = '/home/Image_Processing/July/binaries_w7/'
csv_dir = '/home/Image_Processing/July/csv_w7/'

img_filenames = sorted(os.listdir(img_dir))
csv_filenames = sorted(os.listdir(csv_dir))

for img_filename, csv_filename in zip(img_filenames, csv_filenames):
    img = cv2.imread(img_dir + img_filename)
    # bmg = cv2.imread(bin_dir + bin_filename, cv2.IMREAD_GRAYSCALE)
    dataset = rasterio.open(img_dir + img_filename)
    binary = np.zeros(img.shape[:2], dtype=np.uint8)

    with open(csv_dir + csv_filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            x, y = [np.float128(s) for s in row]
            i, j = dataset.index(x, y, op=round)
            try:
                binary[int(i), int(j)] = 255
            except IndexError:
                print(csv_filename.split('.')[0], "IndexError", x,y)
                continue

    # print(csv_filename, np.all(binary == bmg))
    cv2.imwrite(bin_dir + csv_filename.split('.')[0] + '.tif', binary)
    print(csv_filename.split('.')[0])
