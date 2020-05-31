import os
import argparse
import csv
import re
import numpy as np
from scipy import stats
from keras import optimizers

from imutils import paths as imutils_paths

from utils.preprocessors import \
    ResizePreprocessor, DecorrstretchPreprocessor, ContourPreprocessor, DensityPreprocessor, CountPreprocessor, SubImagePreprocessor
from utils.postprocessors import AggregateLocalCounts
from utils.data_loaders import DataLoader
from utils.neural_nets.tassel_net import TasselNet


def read_parameters_csv(filename):
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        parameters = {}
        for row in reader:
            if row[0] in ['img_dimensions', 'sub_img_dimensions', 'strides', 'agg_strides']:
                parameters[row[0]] = tuple([int(s) for s in re.findall(r'[0-9]+', row[1])])
            elif row[0] in ['point_radius', 'blur_size', 'blur_sd']:
                parameters[row[0]] = int(row[1])
            elif row[0] in ['training_set_fraction']:
                parameters[row[0]] = float(row[1])

    return parameters


def load_dataset_for_training_modes(parameters):

    # Define Preprocessors
    resizor = ResizePreprocessor(*parameters.get('img_dimensions'))
    decorrstretcher = DecorrstretchPreprocessor()
    contours = ContourPreprocessor(parameters.get('point_radius'))
    blurring = DensityPreprocessor(parameters.get('blur_size'), parameters.get('blur_sd'))
    counter = CountPreprocessor()
    img_cropper_for_training = SubImagePreprocessor(parameters.get('sub_img_dimensions'), parameters.get('strides'))
    img_cropper_for_predictions = SubImagePreprocessor(parameters.get('sub_img_dimensions'), parameters.get('agg_strides'))

    # Define DataLoaders
    train_images_loader = DataLoader(preprocessors=[decorrstretcher, resizor, img_cropper_for_training])
    train_counts_loader = DataLoader(preprocessors=[resizor, contours, blurring, img_cropper_for_training, counter], grayscale=True)
    test_images_loader = DataLoader(preprocessors=[decorrstretcher, resizor, img_cropper_for_predictions])
    test_counts_loader = DataLoader(preprocessors=[resizor, contours, blurring, img_cropper_for_predictions, counter], grayscale=True)

    # Train-Test Split
    image_paths = np.sort(np.fromiter(imutils_paths.list_images(args.get("images")), dtype='<U128'))
    binary_paths = np.sort(np.fromiter(imutils_paths.list_images(args.get("binaries")), dtype='<U128'))
    L1, L2 = len(image_paths), len(binary_paths)
    assert L1 == L2, "number of images ({}) does not equal number of binaries ({})".format(L1, L2)
    assert L1 > 0, "empty directory: {}".format(args.get("images"))
    assert L2 > 0, "empty directory: {}".format(args.get("binaries"))
    train_indices = np.random.choice(range(L1), size=round(L1 * parameters.get('training_set_fraction')), replace=False)
    test_indices = np.delete(np.arange(L1), train_indices)

    # Load Images
    print("[INFO]: loading images...")
    train_images = train_images_loader.load_continuous(image_paths[train_indices], verbose=-1)
    test_images = test_images_loader.load_discontinuous(image_paths[test_indices], verbose=-1)
    print("[INFO]: loading binaries...")
    train_counts = train_counts_loader.load_continuous(binary_paths[train_indices], verbose=-1).reshape(-1,1)
    test_counts = np.expand_dims(test_counts_loader.load_discontinuous(binary_paths[test_indices], verbose=-1), axis=2)

    return train_images, train_counts, test_images, test_counts


def load_dataset_for_test_mode(parameters):

    # Define Preprocessors
    resizor = ResizePreprocessor(*parameters.get('img_dimensions'))
    decorrstretcher = DecorrstretchPreprocessor()
    contours = ContourPreprocessor(parameters.get('point_radius'))
    blurring = DensityPreprocessor(parameters.get('blur_size'), parameters.get('blur_size'))
    counter = CountPreprocessor()
    img_cropper_for_predictions = SubImagePreprocessor(parameters.get('sub_img_dimensions'), parameters.get('agg_strides'))

    # Define DataLoaders
    images_loader = DataLoader(preprocessors=[decorrstretcher, resizor, img_cropper_for_predictions])
    counts_loader = DataLoader(preprocessors=[resizor, contours, blurring, img_cropper_for_predictions, counter], grayscale=True)

    image_paths = np.sort(np.fromiter(imutils_paths.list_images(args.get("images")), dtype='<U128'))
    binary_paths = np.sort(np.fromiter(imutils_paths.list_images(args.get("binaries")), dtype='<U128'))
    L1, L2 = len(image_paths), len(binary_paths)
    assert L1 == L2, "number of images ({}) does not equal number of binaries ({})".format(L1, L2)
    assert L1 > 0, "empty directory: {}".format(args.get("images"))
    assert L2 > 0, "empty directory: {}".format(args.get("binaries"))

    images = images_loader.load_discontinuous(image_paths, verbose=-1)
    counts = np.expand_dims(counts_loader.load_discontinuous(binary_paths, verbose=-1), axis=2)

    return images, counts


def load_dataset_for_predict_mode(parameters):

    # Define Preprocessors
    resizor = ResizePreprocessor(*parameters.get('img_dimensions'))
    decorrstretcher = DecorrstretchPreprocessor()
    img_cropper_for_predictions = SubImagePreprocessor(parameters.get('sub_img_dimensions'), parameters.get('agg_strides'))

    # Define DataLoaders
    images_loader = DataLoader(preprocessors=[decorrstretcher, resizor, img_cropper_for_predictions])

    image_paths = np.sort(np.fromiter(imutils_paths.list_images(args.get("images")), dtype='<U128'))
    assert len(image_paths) > 0, "empty directory: {}".format(args.get("images"))
    images = images_loader.load_discontinuous(image_paths, verbose=-1)

    return images


def main():
    if mode in [0, 1]:  # training modes
        if mode == 0:
            if overlapping_aggregation:
                agg_strides = strides
            else:
                agg_strides = sub_img_dimensions

            assert tuple(map(lambda x,y: x % y, img_dimensions, sub_img_dimensions)) == (0, 0), \
                "sub_img_dimensions do not divide img_dimensions"
            assert tuple(map(lambda x,y: x % y, sub_img_dimensions, strides)) == (0, 0), "strides do not divide sub_img_dimensions"
            assert point_radius in range(11), "invalid point_radius; choose value between 0-10 (inclusive)"
            assert blur_size % 2 == 1, "blur size must be odd"

            processing_parameters = {
                'img_dimensions':img_dimensions,
                'sub_img_dimensions':sub_img_dimensions,
                'strides':strides,
                'agg_strides':agg_strides,
                'point_radius':point_radius,
                'blur_size':blur_size,
                'blur_sd':blur_sd,
                'training_set_fraction':training_set_fraction
            }

            tasselnet = TasselNet()
            tasselnet.build(architecture=nn_architecture, input_shape=tuple((*sub_img_dimensions, 3)))
        else:
            parameters_csv = re.sub(r'model_(.*?).hdf5', r'parameters_\1.csv', args.get('net'))
            processing_parameters = read_parameters_csv(parameters_csv)
            tasselnet = TasselNet(args.get("net"))

        # Train Model
        train_images, train_counts, test_images, test_counts = load_dataset_for_training_modes(processing_parameters)

        for params in training_parameters:
            tasselnet.train(train_images, train_counts, processing_parameters, models_dir=args.get("models_dir"), training_parameters=params)


        # Display Model Accuracy
        aggregator = AggregateLocalCounts(img_shape=processing_parameters.get("img_dimensions"),
                                          sub_img_shape=processing_parameters.get("sub_img_dimensions"),
                                          strides=processing_parameters.get("strides"),
                                          point_radius=processing_parameters.get("point_radius"))

        MAE, predictions, counts = tasselnet.test(test_images, test_counts, aggregator)
        print("Summary MAE Statistics: ", stats.describe(MAE))
        print(sum(predictions) - sum(counts))

    elif mode == 2:  # testing mode
        parameters_csv = re.sub(r'model_(.*?).hdf5', r'parameters_\1.csv', args.get('net'))
        processing_parameters = read_parameters_csv(parameters_csv)
        aggregator = AggregateLocalCounts(img_shape=processing_parameters.get("img_dimensions"),
                                          sub_img_shape=processing_parameters.get("sub_img_dimensions"),
                                          strides=processing_parameters.get("strides"),
                                          point_radius=processing_parameters.get("point_radius"))

        images, counts = load_dataset_for_test_mode(processing_parameters)
        tasselnet = TasselNet(args.get("net"))
        MAE, predictions, counts = tasselnet.test(images, counts, aggregator)
        print("Summary MAE Statistics: ", stats.describe(MAE))
        print(sum(predictions) - sum(counts))

    else:  # predicting mode
        parameters_csv = re.sub(r'model_(.*?).hdf5', r'parameters_\1.csv', args.get('net'))
        processing_parameters = read_parameters_csv(parameters_csv)
        aggregator = AggregateLocalCounts(img_shape=processing_parameters.get("img_dimensions"),
                                          sub_img_shape=processing_parameters.get("sub_img_dimensions"),
                                          strides=processing_parameters.get("strides"),
                                          point_radius=processing_parameters.get("point_radius"))

        images = load_dataset_for_predict_mode(processing_parameters)
        tasselnet = TasselNet(args.get("net"))
        predictions = tasselnet.predict(images, aggregator)
        print(predictions)


# --------------------------------------------------------------------------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", default="dataset/Week3/images/", required=False, help="path to images")
ap.add_argument("-b", "--binaries", default="dataset/Week3/binaries/", required=False, help="path to binaries")
ap.add_argument("-s", "--models-dir", default="models/", required=False, help="path to models directory")
ap.add_argument("-n", "--net", required=False, help="path to saved model")
ap.add_argument("-m", "--mode", default='train-new-model', required=False, help="runtime mode")
args = vars(ap.parse_args())

# Parameter Definitions:----------------------------------------------------------------------------------------------------------------------

modes = {'train-new-model':0, 'train-saved-model':1, 'test-saved-model':2, 'predict-counts':3}
mode = modes[args.get('mode')]

if mode > 0 and (not os.path.exists(args.get('net', '')) or args.get('net').split('.')[-1] != 'hdf5'):
    raise Exception("mode {} requires valid saved model".format(args.get('mode')))

# Dataset Parameters:-------------------------------------------------------------------------------------------------------------------------

img_dimensions = (384,1600)
sub_img_dimensions = (32,32)
strides = (8,8)
point_radius = 3
blur_size = 7
blur_sd = 2
training_set_fraction = 3 / 4
overlapping_aggregation = True

# Neural Net Parameters:----------------------------------------------------------------------------------------------------------------------

nn_architecture = 'alexnet'
training_parameters = [
    {
        'batch_size':128,
        'optimizer':optimizers.SGD(lr=0.01, momentum=0.1),
        'epochs':7
    },
    {
        'batch_size':128,
        'optimizer':optimizers.SGD(lr=0.001, momentum=0.1),
        'epochs':3
    }
]

# --------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
