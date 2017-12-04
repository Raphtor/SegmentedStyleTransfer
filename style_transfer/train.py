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

from style_net_net import StyleNet
from vgg import Vgg16

import image_utils


dataset = '../data/train'
img_dim = 256
batch_size = 16
learning_rate = 1e-3
epochs = 1

base_weight = 1e5
style_weight = 1e10
log_interval = 500
def train(args):
	
	# COCO 2014 train dataset, reshape to keep same size
	train_dataset = datasets.ImageFolder(dataset, transforms.Compose([
															transforms.Scale(img_dim),
															transforms.CenterCrop(img_dim),
															transforms.ToTensor(),
															transforms.Lambda(lambda x: x.mul(255))
														]))

	# data loader for traning images
	train_loader = DataLoader(train_dataset, batch_size=batch_size)

	# prepare network to detect features
	vgg = Vgg16(requires_grad=False).cuda()

	# prepare network to apply style
	style_net = StyleNet().cuda()
	optimizer = Adam(style_net.parameters(), learning_rate)
	mse_loss = torch.nn.MSELoss()

	# read style image and convert to tensor
	style_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.mul(255))
	])
	style = image_utils.load(args.style_image)
	style = style_transform(style)
	style = style.repeat(batch_size, 1, 1, 1)
	style = style.cuda()		

	# get Gram matrix from style features
	style_v = Variable(style).cuda()
	style_v = normalize_batch(style_v)
	features_style = vgg(style_v)
	gram_style = [gram_matrix(y) for y in features_style]


	# main training loop over epochs and batches
	for e in range(epochs):
		style_net.train()
		agg_base_loss = 0.
		agg_style_loss = 0.
		count = 0
		for batch_id, (x, _) in enumerate(train_loader):
			n_batch = len(x)
			count += n_batch
			optimizer.zero_grad()

			# raw input image
			x = Variable(x).cuda()
			
			# stylized image
			y = style_net(x)

			# pre-process images according to ImageNet parameters
			y = normalize_batch(y)
			x = normalize_batch(x)

			# detect features in raw and stylized images
			features_y = vgg(y)
			features_x = vgg(x)

			base_loss = base_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

			# determine loss for style based on Gram matrices
			style_loss = 0.
			for ft_y, gm_s in zip(features_y, gram_style):
				gm_y = gram_matrix(ft_y)
				style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
			style_loss *= style_weight

			total_loss = base_loss + style_loss
			total_loss.backward()
			optimizer.step()

			agg_base_loss += base_loss.data[0]
			agg_style_loss += style_loss.data[0]

			if (batch_id + 1) % log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\tbase: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
					time.ctime(), e + 1, count, len(train_dataset),
								  agg_base_loss / (batch_id + 1),
								  agg_style_loss / (batch_id + 1),
								  (agg_base_loss + agg_style_loss) / (batch_id + 1)
				)
				print(mesg)

	# save model
	style_net.eval()
	style_net.cpu()

	save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
		base_weight) + "_" + str(style_weight) + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(style_net.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)



def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram


def normalize_batch(batch):
	# normalize using imagenet mean and std
	mean = batch.data.new(batch.data.size())
	std = batch.data.new(batch.data.size())
	mean[:, 0, :, :] = 0.485
	mean[:, 1, :, :] = 0.456
	mean[:, 2, :, :] = 0.406
	std[:, 0, :, :] = 0.229
	std[:, 1, :, :] = 0.224
	std[:, 2, :, :] = 0.225
	batch = torch.div(batch, 255.0)
	batch -= Variable(mean)
	batch = batch / Variable(std)
	return batch





def main():
    parser = argparse.ArgumentParser(description='Binary-classification with BCE')

    # require either load or save
    parser.add_argument('--style-image', type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")

    parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()