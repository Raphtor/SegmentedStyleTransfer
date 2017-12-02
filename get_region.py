
import cv2
import numpy as np


def get_region(img, x,y):
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eimg = cv2.Canny(gimg, 100,200)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(eimg,kernel,iterations = 1)
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    ff1 = cv2.floodFill(dilation.copy(), mask, (y,x),255)
    mask = (ff1[1] - dilation) / 255
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, 2)
    return mask
