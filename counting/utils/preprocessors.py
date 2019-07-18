import cv2
from itertools import product
import numpy as np
from numpy.linalg import inv, eig


class ResizePreprocessor:
    def __init__(self, height, width, interpolation=cv2.INTER_AREA):
        """
        :param width: Image width
        :param height: Image height
        :param interpolation: Interpolation algorithm
        """
        self.height = height
        self.width = width
        self.interpolation = interpolation

    # Method: Used to resize the image to a fixed size (ignoring the aspect ratio)
    def preprocess(self, image):
        """
        :param image: Image
        :return: Re-sized image
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)


class DecorrstretchPreprocessor:

    def __decorrstretch(self, img):
        a = img.reshape((-1,3)).T.astype(np.float64)
        m = a.mean(1).reshape(3,1)

        Cov = np.cov(a)
        sigma = np.diag(np.sqrt(Cov.diagonal()))
        # cov_eigenvalues, cov_eigenvectors = np.linalg.eig(Cov)

        Cor = inv(sigma) @ Cov @ inv(sigma)
        cor_eigenvalues, cor_eigenvectors = eig(Cor)

        S = np.diag(1 / np.sqrt(cor_eigenvalues))
        T = sigma @ cor_eigenvectors @ S @ cor_eigenvectors.T @ inv(sigma)

        b = ((T @ (a - m)) + m).T.reshape(img.shape)
        c = 255 * (b - b.min(axis=(0,1))) / (b.max(axis=(0,1)) - b.min(axis=(0,1)))

        return c.astype(np.uint8)

    def preprocess(self, image):
        return self.__decorrstretch(image)


class ContourPreprocessor:
    def __init__(self, point_radius=0):
        self.point_radius = point_radius

    def preprocess(self, image):
        binary = np.zeros(image.shape)
        _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                cX, cY = c.mean(0).flatten().astype(int)
            else:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(binary, (cX, cY), self.point_radius, 1, -1)

        return binary


class DensityPreprocessor:
    def __init__(self, k_size, sd):
        if type(k_size) == int:
            self.k_size = (k_size, k_size)
        elif type(k_size) == tuple:
            self.k_size = k_size
        self.sd = sd

    def preprocess(self, image):
        return np.float128(cv2.GaussianBlur(image, self.k_size, self.sd))


class CountPreprocessor:
    def __init__(self, point_radius=0):
        radius_to_pixels = {0:1, 1:5, 2:13, 3:29, 4:49, 5:81, 6:113, 7:149, 8:197, 9:253, 10:317}
        self.divisor = radius_to_pixels[point_radius]

    def preprocess(self, sub_images):
        return sub_images.sum(axis=(1,2)) / self.divisor


class SubImagePreprocessor:
    # Constructor
    def __init__(self, sub_img_shape=(32,32), stride=8):
        """
        :param sub_width: sub-image width
        :param sub_height: sub-image height
        :param stride: vertical and horizontal stride length
        """
        self.sub_img_shape = sub_img_shape
        self.stride = stride

    def preprocess(self, image):
        """
        :param image
        :return numpy array of subimages of size sub_img_shape
        """
        indices = product(range(0, image.shape[0] - self.sub_img_shape[0] + 1, self.stride),
                          range(0, image.shape[1] - self.sub_img_shape[1] + 1, self.stride))

        if len(image.shape) == 3:
            return np.array([image[i:i + self.sub_img_shape[0], j:j + self.sub_img_shape[1], :] for i, j in indices])
        elif len(image.shape) == 2:
            return np.array([image[i:i + self.sub_img_shape[0], j:j + self.sub_img_shape[1]] for i, j in indices])
