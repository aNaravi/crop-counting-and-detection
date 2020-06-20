

# Crop Counting and Detection

A project to count and detect plants or crops in agricultural fields. Currently, only the Counting module is functional. 
Includes an implementation of the following papers:

[TasselNet: counting maize tassels in the wild via local counts regression network](https://plantmethods.biomedcentral.com/track/pdf/10.1186/s13007-017-0224-0)


# Getting Started

*Using the Docker image, which has all the requirements built-in, is recommended.*
Alternatetively, the list of python packages needed is given in `requirements.txt`.


## Docker

Make sure that both Docker (v. 19.03.11) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) are installed before running the following command:

    $ docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /path/to/repo:/home/crop-counting-and-detection anaravi/crop-counting-and-detection bash


## Usage

Illustration with the [Maize Tassel Counting Dataset](https://github.com/poppinace/mtc):

[ `cd /path/to/repo` ]

Extract the images into `datasets/Maize_Tassel_Counting_Dataset/Images`

Extract the annotations into `datasets/Maize_Tassel_Counting_Dataset/Annotations`

Create Binary Images from Matlab Annotations:

    $ python annotating/matlab_annotations_to_binary.py -i datasets/Maize_Tassel_Counting_Dataset/Images -a datasets/Maize_Tassel_Counting_Dataset/Annotations

Train a new Model:

    $ python counting/counting.py -i datasets/Maize_Tassel_Counting_Dataset/Images -b datasets/Maize_Tassel_Counting_Dataset/Binaries


# Dataset

The dataset can be classified into the following:

1.  Images: Unedited RGB images that are processed to produce counts.
2.  Binaries: Binary images with a white pixel at the location of each crop and black pixels otherwise.
3.  Annotations: Either .mat files containing locations of the plants in the Images or RGB images edited with circles to highlight the plants.

Each Image must have a corresponding Annotation and Binary with the same name. The Images and Binaries are fed into the models for training while the Annotations are used to create the Binaries.

For Images that have .mat files containing the locations of their plants, use the `matlab_annotations_to_binary.py` script to produce the corresponding Binaries.

    $ python annotating/matlab_annotations_to_binary.py -h
    usage: matlab_annotations_to_binary.py [-h] -i IMAGES -a ANNOTATIONS [-b BINARIES]
    
    optional arguments:
      -h, --help      show this help message and exit
      -i IMAGES       path to images
      -a ANNOTATIONS  path to annotations
      -b BINARIES     binaries directory name

For Images that do not have .mat files, annotated images and their corresponding Binaries must be created manually using the `annotate_images.py` script.


# Runtime Modes

There are 4 modes that the `counting.py` script can run in:

    $ python counting/counting.py -h
    usage: counting.py [-h] -i IMAGES [-b BINARIES] [-s SAVE] [-m MODE] [-n NEURAL-NET]
    
    optional arguments:
      -h, --help     show this help message and exit
      -i IMAGES      path to images
      -b BINARIES    path to binaries
      -s SAVE        path to save new models (default: crop-counting-and-detection/counting/models)
      -m MODE        runtime mode 
                     choices are (case-sensitive): TrainNewModel (default), TrainSavedModel, TestSavedModel, PredictCounts
      -n NEURAL-NET  path to an existing trained model 
                     parameters csv must be in the same folder with the same timestamp

1.  `TrainNewModel` is used to train a new model and save it along with the parameters required to use that model in the future.
2.  `TrainSavedModel` can be used to train a saved model on a new set of images.
3.  `TestSavedModel` is used to test a saved model on a new set of images.
4.  `PredictCounts` is used to run a saved model on any set of images to give the crop-count in them.

Apart from `PredictCounts`, all other modes require the paths to both Images and Binaries.

When using a saved model, the parameters that were used while training it are required in .csv format. This csv file is automatically generated and saved in the `TrainNewModel` and `TrainSavedModel` modes.

