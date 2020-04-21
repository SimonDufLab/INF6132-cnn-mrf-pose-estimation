import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
from torch.utils import data
from matplotlib import colors
from utils import mkdir


# Folder containing 4 .npz files for training and test set
data_folder = "./data/"


def to_dataloader(X, y, batch_size=10):
    # Numpy arrays to pytorch DataLoader
    X, y = torch.Tensor(X), torch.Tensor(y)

    # Permute so that we have (batch, channels, height, width)
    X, y = X.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)

    dataset = data.TensorDataset(X, y)
    dataloader = data.DataLoader(dataset, batch_size=batch_size)

    return dataloader


def load_train_data(path=data_folder):
    print(f"Loading train data from : {path}... ", end="")

    X_train = np.load(f"{path}x_train_flic.npz")["arr_0"]
    y_train = np.load(f"{path}y_train_flic.npz")["arr_0"]

    print("Done.")

    return X_train, y_train


def load_test_data(path=data_folder):
    print(f"Loading test data from : {path}... ", end="")

    X_test = np.load(f"{path}x_test_flic.npz")["arr_0"]
    y_test = np.load(f"{path}y_test_flic.npz")["arr_0"]

    print("Done.")

    return X_test, y_test


def load_data(path=data_folder):
    # Load train and test data
    X_train, y_train = load_train_data(path)
    X_test, y_test = load_test_data(path)

    return X_train, y_train, X_test, y_test


def viz_sample(image, heatmap, name, save_dir):
    # Vizualise single image and heatmap target using
    # matplotlib
    image = resize(image, (60, 90))
    joint_colors = ("red", "green", "blue", "yellow", "purple",
                    "orange", "black", "white", "cyan", "darkblue")

    # Iterate on the heatmap for each joint
    for i in range(heatmap.shape[2]):
        plt.imshow(image)
        cs = [(0, 0, 0, 0), joint_colors[i]]
        color_map = colors.LinearSegmentedColormap.from_list("cmap", cs)

        heatmap_data = heatmap[:, :, i]
        #heatmap_data = resize(heatmap_data, (480, 720))

        # Mask image to show only joint location
        heatmap_data[heatmap_data < 0.01] = 0
        plt.imshow(heatmap_data, cmap=color_map)

        save_path = f"{save_dir}/images/joint_{i}/{name}.png"
        mkdir(save_path, path_is_file=True)
        plt.savefig(save_path)


def main():
    X_train, y_train, X_test, y_test = load_data(data_folder)
    train = to_dataloader(X_train, y_train)
    test = to_dataloader(X_test, y_test)

    image_number = 10

    # Vizualise first sample of train dataset
    viz_sample(train.dataset[image_number][0], train.dataset[image_number][1])

    # Vizualise first sample of test dataset
    viz_sample(test.dataset[image_number][0], test.dataset[image_number][1])


if __name__ == "__main__":
    main()
