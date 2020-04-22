# INF6132-cnn-mrf-pose-estimation

Pytorch implementation of [Joint Training of a Convolutional Network and aGraphical Model for Human Pose Estimation](http://papers.nips.cc/paper/5573-joint-training-of-a-convolutional-network-and-a-graphical-model-for-human-pose-estimation).

## Download processed data

Download a small processed subset of the FLIC dataset [here](https://drive.google.com/open?id=1cx62R_H5j3f-ZTXREFlYpESHf3R6vc1Q). This subset contains 500 training images and 100 testing images with their heatmap targets. The processing was done using the `raw_data.py` script, very closely based on [this] script. Place the 4 .npz files in a `data/` folder inside this directory, e.g. :

    ```
    ├── ...
    ├── data                    # data folder
    │   ├── x_test_flic.npz
    │   ├── x_train_flic.npz
    │   ├── y_test_flic.npz
    │   └── y_train_flic.npz
    └── ...
    ```

## Requirements

- [Pytorch](https://pytorch.org/) with torchvision
- [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
    ```
    pip install pytorch-lightning
    ```
- Matplotlib
    ```
    pip install matplotlib
    ```
- scikit-image
    ```
    pip install scikit-image
    ```

## Problem description

The goal of the project is to estimate the pose of a single person using a single 2D image. The approach is based on the paper "Joint Training of a Convolutional Network and aGraphical Model for Human Pose Estimation" Tompson et al. [2014]. The model is segmented in two parts. The first one is a part detector, a Convolutional Neural Network (CNN), is used to generate a heatmap of probable locations of each body parts (e.g. left shoulder, right elbow, etc.). The second part is a spatial model which is used to refine the output of the CNN by enforcing the kinematics constraints of the body parts. These two models can then be trained jointly.

## Data

The dataset used in this project is the Frames Labeled In Cinema (FLIC) dataset (Sapp and Taskar [2013]). The dataset contists of 5003 images taken from Hollywood movies. The resolution of the images is 720x480 pixels with 3 channels (colored). Each image is labeled with body parts joint location: left and right wrists, left and right elbow, left and right shoulder, left and right hips, nose and torso. 80% of data is used for training (3987 images) and 20% is used for testing (1016 images). In the original dataset, left and right eyes are also annotated, but they are not used in our project as we are not interested in such precise information.

The location of each joint (body part) is given by a heatmap with a resolution of 90x60 pixels with a single channel. Since there are 10 joints, for each image we have a heatmap of 90x60 pixels with 10 channels (one channel per body part). The heatmap value is 0 for every pixel except where the body part is located where a small gaussian is used instead of a single 1 to account for the incertitude of the body part location.

## Model details

### Part detector (CNN)

### Spatial Model (MRF)

## Train the CNN Part-Detector

WORK IN PROGRESS, NOT CURRENTLY WORKING

To train the CNN Part-Detector, simply run :

```
python cnn_part_detector.py
```


## TODO:
Currently the program only contains the Part Detector (CNN) and not the Spatial Model (MRF). I am trying to get a decent output with the CNN (which according to the [this](https://github.com/max-andr/joint-cnn-mrf) implementation we are supposed to be able to get) before starting The MRF.

- Not sure what loss to use and how to implement. The paper mentions that they use a MSE loss. Do they use it directly on the 2d heatmaps? E.g. MSELoss(predicted_heatmap, target_heatmap) with the heatmaps being of dimension [height, width, number_of_joints] (which is in the paper 60x90x10)

- I have not started the spatial model yet (the markov random field). Not entirely sure how to implement it yet. 
