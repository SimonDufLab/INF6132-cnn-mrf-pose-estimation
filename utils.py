import torch
import torchvision
from torchvision import transforms


def resize_batch(input_tensors, h, w):
    # Resize a batch of imges using torchvision transforms
    # We need to convert Tensors to PIL Images and back
    final_output = None
    batch_size, channel, height, width = input_tensors.shape
    input_tensors = torch.squeeze(input_tensors, 1)

    for img in input_tensors:
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = torchvision.transforms.Resize([h, w])(img_PIL)
        img_PIL = torchvision.transforms.ToTensor()(img_PIL)
        img_PIL = img_PIL.unsqueeze(0)
        if final_output is None:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    return final_output
