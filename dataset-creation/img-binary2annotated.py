import os
import numpy as np
import cv2
import re

img_dir = '/home/Image_Processing/July/images_w4/'
bin_dir = '/home/Image_Processing/July/binaries_w4/'
ann_dir = '/home/Image_Processing/July/annotated_w4/'

img_filenames = sorted(os.listdir(img_dir))
bin_filenames = sorted(os.listdir(bin_dir))

for img_filename, bin_filename in zip(img_filenames, bin_filenames):
    img = cv2.imread(img_dir + img_filename)
    bmg = cv2.imread(bin_dir + bin_filename, cv2.IMREAD_GRAYSCALE)
    ann = img.copy()

    for (r,c) in np.array(np.where(bmg == 255)).T:
        cv2.circle(ann, (c,r), 3, (0,0,255), -1)

    cv2.imwrite(ann_dir + re.sub(r'.*?_([0-9]{3})', r'annotated_\1', img_filename), ann)

