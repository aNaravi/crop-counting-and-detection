import os
import cv2
import numpy as np


class DataLoader:
    # Method: Constructor
    def __init__(self, preprocessors=None):
        """
        :param preprocessors: List of image preprocessors
        """
        if self.preprocessors is None:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors

    # Method: Used to load a list of images for pre-processing
    def load_continuous(self, image_paths, verbose=-1):
        """
        :param image_paths: List of image paths
        :param verbose: Parameter for printing information to console
        :return: numpy array of preprocessed images
        """
        for i, image_path in enumerate(image_paths):
            label = image_path.split(os.path.sep)[-2]
            if image_path.split(os.path.sep)[-2] == 'binaries':
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_path)

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            if i == 0:
                data = image
            else:
                data = np.append(data, image, axis=0)

            if verbose > 0 and (i + 1) % verbose == 0:
                print('[INFO]: ' + label + ' Processed {}/{}'.format(i + 1, len(image_paths)))

        return data

    def load_discontinuous(self, image_paths, verbose=-1):
        """
        :param image_paths: List of image paths
        :param verbose: Parameter for printing information to console
        :return: numpy array of preprocessed images
        """
        data = []
        for i, image_path in enumerate(image_paths):
            label = image_path.split(os.path.sep)[-2]
            if image_path.split(os.path.sep)[-2] == 'binaries':
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_path)

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)

            if verbose > 0 and (i + 1) % verbose == 0:
                print('[INFO]: ' + label + ' Processed {}/{}'.format(i + 1, len(image_paths)))

        return np.array(data)
