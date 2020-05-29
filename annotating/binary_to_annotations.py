import numpy as np
import cv2
import re
import argparse
from imutils import paths as im_paths


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", default=".", required=False, help="dataset directory")
args = vars(ap.parse_args())

dir_images = args.get('dir') + '/Images/'
dir_binaries = args.get('dir') + '/Binaries/'

images = sorted(im_paths.list_files(dir_images, validExts='jpg'))
binaries = sorted(im_paths.list_files(dir_binaries, validExts='png'))

for img_filename, bmg_filename in zip(images, binaries):
    bmg = cv2.imread(bmg_filename, cv2.IMREAD_GRAYSCALE)
    ann = cv2.imread(img_filename)

    for c, r in np.array(np.where(bmg == 255)).T:
        cv2.circle(ann, (r, c), 17, (0,0,255), -1)

    cv2.imwrite(re.sub(r'Images', 'Annotations', img_filename), ann)  # re.sub(r'.*?_([0-9]{3})', r'annotated_\1', img_filename)

