import numpy as np
import cv2
import os
import argparse
from imutils import paths as im_paths


ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("-i", metavar="IMAGES", dest="images", required=True, help="path to images")
ap.add_argument("-b", metavar="BINARIES", dest="binaries", required=True, help="path to binaries")
ap.add_argument("-a", metavar="ANNOTATIONS", dest="annotations", required=False, default='Annotations', help="annotations directory name (default: Annotations)")
args = vars(ap.parse_args())

dir_images = os.path.normpath(args.get('images'))
dir_binaries = os.path.normpath(args.get('binaries'))
dir_annotations = dir_images.replace(os.path.basename(dir_images), args.get('annotations'))
os.makedirs(dir_annotations, exist_ok=True)

images = sorted(im_paths.list_files(dir_images, validExts='jpg'))
binaries = sorted(im_paths.list_files(dir_binaries, validExts='png'))

for img_filename, bmg_filename in zip(images, binaries):
    bmg = cv2.imread(bmg_filename, cv2.IMREAD_GRAYSCALE)
    ann = cv2.imread(img_filename)

    for c, r in np.array(np.where(bmg == 255)).T:
        cv2.circle(ann, (r, c), 17, (0,0,255), -1)

    cv2.imwrite(dir_annotations + os.path.sep + os.path.basename(img_filename), ann)

