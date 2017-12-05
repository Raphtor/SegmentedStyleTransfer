import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms


from .style_net import StyleNet
from .vgg import Vgg16
from . import image_utils


def test(base_filename, model_filename, output_filename):
    base_image = image_utils.load(base_filename)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    base_image = base_transform(base_image)
    base_image = base_image.unsqueeze(0)
    base_image = base_image#.cuda()
    base_image = Variable(base_image, volatile=True)

    style_model = StyleNet()
    style_model.load_state_dict(torch.load(model_filename))
    style_model#.cuda()
    
    output = style_model(base_image)
    output = output.cpu()
    output_data = output.data[0]
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Binary-classification with BCE')

    # require either load or save
    parser.add_argument('--base-image', type=str, required=True,
                                  help="path to base-image")

    parser.add_argument('--model', type=str, required=True,
                                  help="path to saved model file")

    parser.add_argument("--output-images", type=str, required=True,
                                  help="path to where image should be saved")

    args = parser.parse_args()
    test(args.base_image, args.model, args.output_images)

if __name__ == "__main__":
    main()