import torch
import torchvision
from torchvision import transforms
import os


def resize_batch(input_tensors, h, w):
    # Resize a batch of imges using torchvision transforms
    # We need to convert Tensors to PIL Images and back
    final_output = None
    batch_size, channel, height, width = input_tensors.shape
    input_tensors = torch.squeeze(input_tensors, 1)

    for img in input_tensors:
        multi_chan_img = None
        for i in range(channel):
            img_PIL = transforms.ToPILImage()(img[i].unsqueeze(0))
            img_PIL = torchvision.transforms.Resize([h, w])(img_PIL)
            img_PIL = torchvision.transforms.ToTensor()(img_PIL)
            if multi_chan_img is None:
                multi_chan_img = img_PIL
            else:
                multi_chan_img = torch.cat((multi_chan_img, img_PIL), 0)

        multi_chan_img = multi_chan_img.unsqueeze(0)
        if final_output is None:
            final_output = multi_chan_img
        else:
            final_output = torch.cat((final_output, multi_chan_img), 0)
    return final_output


def mkdir(path, path_is_file=False):
    if path_is_file:
        path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)
