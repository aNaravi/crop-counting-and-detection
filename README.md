
# Table of Contents

1.  [Crop Counting and Detection](#org58252b9)
2.  [Getting Started](#orge4fb336)
    1.  [Prerequisites](#org001fda9)
    2.  [Docker](#orgca8f0c4)
    3.  [Usage](#orgeeaab60)
3.  [Workflow](#orge20e923)
    1.  [Using the Maize Dataset:](#orgd10c4ff)


<a id="org58252b9"></a>

# Crop Counting and Detection

A project to count and detect plants or crops in agricultural fields which includes an implementation of the following paper:

[TasselNet: counting maize tassels in the wild via local counts regression network](https://plantmethods.biomedcentral.com/track/pdf/10.1186/s13007-017-0224-0)


<a id="orge4fb336"></a>

# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


<a id="org001fda9"></a>

## Prerequisites

-   The list of python packages is in \`requirement.txt\`.
-   OpenCV (v. 3.4.5)

Alternatively, one may use the Docker image that contains all the required packages.


<a id="orgca8f0c4"></a>

## Docker

Make sure you have both both docker (v. 19.03.11) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed before running the following command:

    $ docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /path/to/repo:/home/crop-counting anaravi/crop-counting-and-detection bash


<a id="orgeeaab60"></a>

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


<a id="orge20e923"></a>

# Workflow


<a id="orgd10c4ff"></a>

## Using the [Maize Dataset](https://github.com/poppinace/mtc):

1.  Extract the images into \`datasets/Maize<sub>Tassel</sub><sub>Counting</sub><sub>Dataset</sub>/Images\`

2.  Extract the annotations into \`datasets/Maize<sub>Tassel</sub><sub>Counting</sub><sub>Dataset</sub>/Annotations\`

3.  Get Binary Images from Matlab Annotations

    $ python annotating/matlab_annotations_to_binary.py -i datasets/Maize_Tassel_Counting_Dataset/Images -b datasets/Maize_Tassel_Counting_Dataset/Annotations
    #+end_src sh
    
    4. Train a new Model
    
    #+begin_src sh
    $ python counting/counting.py -i datasets/Maize_Tassel_Counting_Dataset/Images -b datasets/Maize_Tassel_Counting_Dataset/Binaries

