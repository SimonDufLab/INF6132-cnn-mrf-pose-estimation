# INF6132-cnn-mrf-pose-estimation

Pytorch implementation of [Joint Training of a Convolutional Network and aGraphical Model for Human Pose Estimation](http://papers.nips.cc/paper/5573-joint-training-of-a-convolutional-network-and-a-graphical-model-for-human-pose-estimation).

## Download processed data

Download a small processed subset of the FLIC dataset [here](https://drive.google.com/open?id=1cx62R_H5j3f-ZTXREFlYpESHf3R6vc1Q). This subset contains 500 training images and 100 testing images with their heatmap targets. The processing was done using the ```raw_data.py``` script, very closely based on [this] script. Place the 4 .npz files in a ```data/``` folder inside this directory, e.g. :

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
