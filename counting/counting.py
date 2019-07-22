import argparse
import numpy as np
# import matplotlib.pyplot as plt
# import cv2
from imutils import paths as imutils_paths
from keras import optimizers
from utils.preprocessors import ResizePreprocessor, DecorrstretchPreprocessor, ContourPreprocessor, \
    DensityPreprocessor, CountPreprocessor, SubImagePreprocessor
from utils.data_loaders import DataLoader
from utils.neural_nets.tassel_net import TasselNet


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", default="dataset/Week6/originals", required=False, help="path to images")
ap.add_argument("-b", "--binaries", default="dataset/Week6/binaries/", required=False, help="path to binaries")
ap.add_argument("-m", "--models", default="models/", required=False, help="model's save path")
args = vars(ap.parse_args())

img_dimensions = (384,1600)
sub_img_dimensions = (64,64)
stride = 8
point_radius = 3
blur = (7,2)

resizor = ResizePreprocessor(*img_dimensions)
decorrstretcher = DecorrstretchPreprocessor()
contours = ContourPreprocessor(point_radius)
blurring = DensityPreprocessor(*blur)
counter = CountPreprocessor()
img_cropper = SubImagePreprocessor(sub_img_dimensions, stride)

images_loader = DataLoader(preprocessors=[resizor, decorrstretcher, img_cropper])
counts_loader = DataLoader(preprocessors=[resizor, contours, blurring, img_cropper, counter])

image_paths = np.sort(np.fromiter(imutils_paths.list_images(args.get("images")), dtype='<U128'))
binary_paths = np.sort(np.fromiter(imutils_paths.list_images(args.get("binaries")), dtype='<U128'))

train_indices = np.random.choice(range(198), size=150, replace=False)
test_indices = np.delete(np.arange(198), train_indices)

print("[INFO]: loading images...")
train_images = images_loader.load_continuous(image_paths[train_indices], verbose=1)
test_images = images_loader.load_discontinuous(image_paths[test_indices], verbose=1)

print("[INFO]: loading binaries...")
train_counts = counts_loader.load_continuous(binary_paths[train_indices], verbose=1).reshape(-1,1)
test_counts = np.expand_dims(counts_loader.load_discontinuous(binary_paths[test_indices], verbose=1), axis=2)

tasselnet = TasselNet()
tasselnet.build(architecture='alexnet', input_shape=tuple((*sub_img_dimensions, 3)))

tasselnet.train(train_images, train_counts,
                save_folder=args.get("models"),
                batch_size=128,
                optimizer=optimizers.SGD(lr=0.01, momentum=0.1),
                epochs=7)
tasselnet.train(train_images, train_counts,
                save_folder=args.get("models"),
                batch_size=128,
                optimizer=optimizers.SGD(lr=0.001, momentum=0.1),
                epochs=3)

MAE, predictions, counts = tasselnet.test(test_images, test_counts,
                                          img_dimensions=img_dimensions,
                                          sub_img_dimensions=sub_img_dimensions,
                                          stride=stride,
                                          point_radius=point_radius)
print(sum(predictions) - sum(counts))
