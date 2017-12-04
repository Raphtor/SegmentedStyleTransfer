
import cv2
import numpy as np


def get_region(gimg, x,y):
    eimg = cv2.Canny(gimg, 100,200)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(eimg,kernel,iterations = 1)
    h, w = gimg.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    ff1 = cv2.floodFill(dilation.copy(), mask, (y,x),255)
    mask = (ff1[1] - dilation) / 255
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, 2)
    return mask


def build_mask(img, points):
    mask = np.zeros(img.shape, np.uint8)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for x, y in points:
        curr_mask = get_region(gimg, x, y)
        mask = np.maximum(mask, curr_mask)

    return mask

if __name__ == '__main__':

    # for testing
    img = cv2.imread('chairs.jpg')
    points = set()
    points.add((215, 150))
    points.add((215, 400))
    mask = build_mask(img, points)

    cv2.imwrite('test.jpg', mask * (255//3))