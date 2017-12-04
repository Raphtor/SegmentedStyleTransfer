import numpy as np 
import cv2

base_img = cv2.imread('../munich.jpg')
stylized_img = cv2.imread('../munich_picasso.jpg')

# TODO: why is sytlized bigger?
stylized_img = stylized_img[:-2, :, :]

x, y, z = base_img.shape 
#mask = np.random.rand(x, y)
#mask = (mask < 0.5) * 1

mask = np.zeros((x, y))
mask[:x//2, :y//2] = 1
mask[x//2:, y//2:] = 1


print(base_img.shape)
rgb_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)#mask[:, None, None] + mask[None, :, None] + mask[None, None, :]

cv2.imwrite('../munich_mask.jpg', mask * 255)

print(base_img.shape)

out_base = np.multiply(base_img, rgb_mask)
out_style = np.multiply(stylized_img, 1 - rgb_mask)
out = out_base + out_style
cv2.imwrite('../munich_segmented_picasso.jpg', out)
