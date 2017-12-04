import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

"""
from style_net import StyleNet
from vgg import Vgg16
import image_utils
"""

def test(args):
    base_image = utils.load_image(args.base_image)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    base_image = base_transform(base_image)
    base_image = base_image.unsqueeze(0)
    base_image = base_image.cuda()
    base_image = Variable(base_image, volatile=True)

    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))
    style_model.cuda()
    
    output = style_model(base_image)
    output = output.cpu()
    output_data = output.data[0]
    utils.save_image(args.output_image, output_data)

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
    train(args)

if __name__ == "__main__":
    main()