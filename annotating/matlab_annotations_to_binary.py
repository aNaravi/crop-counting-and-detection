import cv2
import numpy as np
import scipy.io as spio
import argparse
import os
from imutils import paths as im_paths

ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("-i", metavar="IMAGES", dest="images", required=True, help="path to images")
ap.add_argument("-a", metavar="ANNOTATIONS", dest="annotations", required=True, help="path to annotations")
ap.add_argument("-b", metavar="BINARIES", dest="binaries", required=False, default='Binaries', help="binaries directory name (default: Binaries)")
args = vars(ap.parse_args())

dir_images = os.path.normpath(args.get('images'))
dir_annotations = os.path.normpath(args.get('annotations'))
dir_binaries = dir_images.replace(os.path.basename(dir_images), args.get('binaries'))
os.makedirs(dir_binaries, exist_ok=True)

images = sorted(im_paths.list_files(dir_images, validExts='jpg'))
annotations = sorted(im_paths.list_files(dir_annotations, validExts='mat'))

for i, a in zip(images, annotations):
    img, ann = cv2.imread(i), spio.loadmat(a)
    bmg = np.zeros(img.shape[:2], dtype=np.uint8)

    ann_idx, filename = ann['annotation']['bndbox'][0,0], ann['annotation']['filename'][0,0][0]

    if ann_idx.shape != (0, 0):
        bmg[ann_idx[:,1], ann_idx[:,0]] = 255

    cv2.imwrite(dir_binaries + os.path.sep + filename + ".png", bmg)
