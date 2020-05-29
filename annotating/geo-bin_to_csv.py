import os
import numpy as np
import cv2
import rasterio
import csv

img_dir = '/home/Image_Processing/July/images_w2/'
bin_dir = '/home/Image_Processing/July/binaries_w2/'
csv_dir = '/home/Image_Processing/July/csv_w2_/'

img_filenames = sorted(os.listdir(img_dir))
bin_filenames = sorted(os.listdir(bin_dir))

for img_filename, bin_filename in zip(img_filenames, bin_filenames):
    img = cv2.imread(img_dir + img_filename)
    bmg = cv2.imread(bin_dir + bin_filename, cv2.IMREAD_GRAYSCALE)
    dataset = rasterio.open(img_dir + img_filename) 

    with open(csv_dir + bin_filename.split('.')[0] + '_geotags.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(np.array(dataset.transform * np.array(np.where(bmg == 255))[::-1,:]).T)
