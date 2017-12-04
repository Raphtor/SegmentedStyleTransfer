import torch
from PIL import Image
from torch.autograd import Variable

import numpy as np 
import cv2

def stylize_segments(base_img, stylized_img, mask):
    #base_img = cv2.imread(base_filename)
    #stylized_img = cv2.imread(stylized_filename)

    # TODO: why is sytlized bigger?
    #stylized_img = stylized_img[:-2, :, :]

    #x, y, z = base_img.shape 

    #mask = np.zeros((x, y))
    #mask[:x//2, :y//2] = 1
    #mask[x//2:, y//2:] = 1


    rgb_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    out_base = np.multiply(base_img, rgb_mask)
    out_style = np.multiply(stylized_img, 1 - rgb_mask)
    out = out_base + out_style
    return out


def load(filename, size=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    return img


def save(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)