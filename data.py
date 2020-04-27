import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
from torch.utils import data
from matplotlib import colors
from utils import mkdir


# Folder containing 4 .npz files for training and test set
data_folder = "./data/"


def to_dataloader(X, y, batch_size=10, shuffle = True):
    # Numpy arrays to pytorch DataLoader
    X, y = torch.Tensor(X), torch.Tensor(y)

    # Permute so that we have (batch, channels, height, width)
    X, y = X.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)

    dataset = data.TensorDataset(X, y)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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


def viz_sample(image, heatmap, name=None, save_dir=None, permute = False):
    #Need permutation?
    if permute :
        image = image.permute(1,2,0)
        heatmap = heatmap.permute(1,2,0)

    # Vizualise single image and heatmap target using
    # matplotlib
    image = resize(image, (60, 90, 3))
    joint_colors = ("red", "green", "blue", "yellow", "purple",
                    "orange", "black", "white", "cyan", "darkblue")

    # Iterate on the heatmap for each joint
    plt.imshow(image)
    if name:
        plt.title("Displaying targets: " + name)
    else:
        plt.title("Image with targets on top") # I love title

    for i in range(heatmap.shape[2]):

        cs = [(0, 0, 0, 0), joint_colors[i]]
        color_map = colors.LinearSegmentedColormap.from_list("cmap", cs)

        heatmap_data = heatmap[:, :, i]
        #heatmap_data = resize(heatmap_data, (480, 720))

        # Mask image to show only joint location
        heatmap_data[heatmap_data < 0.01] = 0
        heatmap_data = heatmap_data / np.max(heatmap_data.numpy())
        plt.imshow(heatmap_data, cmap=color_map)

    # If name and save_dir are defined, save to location:
    if name and save_dir:
        save_path = f"{save_dir}/images/{name}.png"
        mkdir(save_path, path_is_file=True)
        plt.savefig(save_path)

    plt.cla()
    plt.clf() ## Clearing axes and current figure should help image saving more efficient.
    #plt.show()


def main():
    X_train, y_train, X_test, y_test = load_data(data_folder)
    train = to_dataloader(X_train, y_train)
    test = to_dataloader(X_test, y_test)

    image_number = 10

    # Vizualise first sample of train dataset
    viz_sample(train.dataset[image_number][0], train.dataset[image_number][1], permute = True)

    # Vizualise first sample of test dataset
    viz_sample(test.dataset[image_number][0], test.dataset[image_number][1], permute = True)


if __name__ == "__main__":
    main()
