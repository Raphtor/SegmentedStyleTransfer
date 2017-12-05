import torch
from PIL import Image
from torch.autograd import Variable

import numpy as np
import cv2

def stylize_segments(base_img, stylized_img, mask):
    out_base = np.multiply(base_img, 1- mask)
    out_style = np.multiply(stylized_img, mask)
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