import cv2
import numpy as np
import scipy.io as spio
import argparse
import os
from imutils import paths as im_paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", default=".", required=False, help="dataset directory")
args = vars(ap.parse_args())

dir_images = args.get('dir') + '/Images/'
dir_annotations = args.get('dir') + '/Annotations/'
dir_binaries = args.get('dir') + '/Binaries/'

os.makedirs(dir_binaries, exist_ok=True)

images = sorted(im_paths.list_files(dir_images, validExts='jpg'))
annotations = sorted(im_paths.list_files(dir_annotations, validExts='mat'))

for i, a in zip(images, annotations):
    img, ann = cv2.imread(i), spio.loadmat(a)
    bmg = np.zeros(img.shape[:2], dtype=np.uint8)

    ann_idx, filename = ann['annotation']['bndbox'][0,0], ann['annotation']['filename'][0,0][0]

    if ann_idx.shape != (0, 0):
        bmg[ann_idx[:,1], ann_idx[:,0]] = 255

    cv2.imwrite(dir_binaries + filename + ".png", bmg) 
