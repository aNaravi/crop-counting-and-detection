
# Table of Contents

1.  [Crop Counting and Detection](#org8d46e6a)
2.  [Getting Started](#orgd49f8b5)
    1.  [Prerequisites](#orgb055bc9)
    2.  [Docker](#orgc093cf1)
    3.  [Usage](#org9bc09f3)
3.  [Workflow](#orgf115521)
    1.  [Illustration of usage with the Maize Dataset:](#org5eac127)


<a id="org8d46e6a"></a>

# Crop Counting and Detection

A project to count and detect plants or crops in agricultural fields which includes an implementation of the following paper:

[TasselNet: counting maize tassels in the wild via local counts regression network](https://plantmethods.biomedcentral.com/track/pdf/10.1186/s13007-017-0224-0)


<a id="orgd49f8b5"></a>

# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


<a id="orgb055bc9"></a>

## Prerequisites

-   The list of python packages is in `requirements.txt`.
-   OpenCV (v. 3.4.5)

Alternatively, one may use the Docker image that contains all the required packages.


<a id="orgc093cf1"></a>

## Docker

Make sure you have both both docker (v. 19.03.11) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed before running the following command:

    $ docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /path/to/repo:/home/crop-counting anaravi/crop-counting-and-detection bash


<a id="org9bc09f3"></a>

## Usage

    $ cd /home/crop-counting
    
    $ python counting/counting.py -h
    usage: counting.py [-h] -i IMAGES -b BINARIES [-s SAVE] [-m MODE]
                       [-n NEURAL-NET]
    
    optional arguments:
      -h, --help     show this help message and exit
      -i IMAGES      path to images
      -b BINARIES    path to binaries
      -s SAVE        path to save new models (default: crop-counting-and-detection/counting/models)
      -m MODE        runtime mode 
                     choices are (case-sensitive): TrainNewModel (default), TrainSavedModel, TestSavedModel, PredictCounts
      -n NEURAL-NET  path to an existing trained model 
                     parameters csv must be in the same folder with the same timestamp
    
    $ python annotating/matlab_annotations_to_binary.py -h
    usage: matlab_annotations_to_binary.py [-h] -i IMAGES -a ANNOTATIONS
                                           [-b BINARIES]
    
    optional arguments:
      -h, --help      show this help message and exit
      -i IMAGES       path to images
      -a ANNOTATIONS  path to annotations
      -b BINARIES     binaries directory name
    
    $ python annotating/binary_to_annotations.py -h
    usage: binary_to_annotations.py [-h] -i IMAGES -b BINARIES [-a ANNOTATIONS]
    
    optional arguments:
      -h, --help      show this help message and exit
      -i IMAGES       path to images
      -b BINARIES     path to binaries
      -a ANNOTATIONS  annotations directory name


<a id="orgf115521"></a>

# Workflow


<a id="org5eac127"></a>

## Illustration of usage with the [Maize Dataset](https://github.com/poppinace/mtc):

1.  Extract the images into `datasets/Maize_Tassel_Counting_Dataset/Images`

2.  Extract the annotations into `datasets/Maize_Tassel_Counting_Dataset/Annotations`

3.  Get Binary Images from Matlab Annotations

    $ python annotating/matlab_annotations_to_binary.py -i datasets/Maize_Tassel_Counting_Dataset/Images -a datasets/Maize_Tassel_Counting_Dataset/Annotations

1.  Train a new Model

    $ python counting/counting.py -i datasets/Maize_Tassel_Counting_Dataset/Images -b datasets/Maize_Tassel_Counting_Dataset/Binaries

