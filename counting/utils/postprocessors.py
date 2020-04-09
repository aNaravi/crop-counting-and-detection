import numpy as np
from itertools import product
from functools import reduce


class AggregateLocalCounts:
    def __init__(self, img_shape=(384,1600), sub_img_shape=(32,32), strides=(8,8), point_radius=0):
        radius_to_pixels = {0:1, 1:5, 2:13, 3:29, 4:49, 5:81, 6:113, 7:149, 8:197, 9:253, 10:317}
        self.img_shape = img_shape
        self.sub_img_shape = sub_img_shape
        self.strides = strides
        self.divisor = radius_to_pixels[point_radius]

    def normalization_array(self):
        array = np.zeros(self.img_shape, np.float128)
        sub_array = np.ones(self.sub_img_shape, np.float128)

        for i, j in product(range(0, self.img_shape[0] - self.sub_img_shape[0] + 1, self.strides[0]),
                            range(0, self.img_shape[1] - self.sub_img_shape[1] + 1, self.strides[1])):
            array[i:i + self.sub_img_shape[0], j:j + self.sub_img_shape[1]] += sub_array

        return array

    def aggregate_local_counts(self, counts):
        array = np.zeros(self.img_shape, np.float128)

        for k, (i, j) in enumerate(product(range(0, self.img_shape[0] - self.sub_img_shape[0] + 1, self.strides[0]),
                                           range(0, self.img_shape[1] - self.sub_img_shape[1] + 1, self.strides[1]))):
            sub_array = np.full(self.sub_img_shape, counts[k] / reduce(lambda x, y: x * y, self.sub_img_shape), dtype=np.float128)
            array[i:i + self.sub_img_shape[0], j:j + self.sub_img_shape[1]] += sub_array

        return (array / self.normalization_array()).sum() / self.divisor

    def aggregate_subimages(self, sub_images):
        array = np.zeros(self.img_shape, np.float128)

        for k, (i, j) in enumerate(product(range(0, self.img_shape[0] - self.sub_img_shape[0] + 1, self.strides[0]),
                                           range(0, self.img_shape[1] - self.sub_img_shape[1] + 1, self.strides[1]))):
            sub_array = sub_images[k]
            array[i:i + self.sub_img_shape[0], j:j + self.sub_img_shape[1]] += sub_array

        return (array / self.normalization_array()).sum() / self.divisor
