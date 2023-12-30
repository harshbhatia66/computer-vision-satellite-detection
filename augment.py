import cv2 as cv
import numpy as np
from dataset.utils.elpv_reader import load_dataset
import skimage
import random

images, proba, types = load_dataset()

class Augment(object):
    def __int__(self, images, proba, types):
        self.images = images
        self.proba = proba
        self.types = types

    # Option 1
    def pepper_noise(self, img):
        peppered_image = skimage.util.random_noise(img, mode="pepper")
        return peppered_image

    # Option 2
    def flip_horizontally(self, img):
        return cv.flip(img, 1)

    # Option 3
    def rotate_90_clockwise(self, img):
        return cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    # Option 4
    def rotate_90_counter_clockwise(self, img):
        return cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Option 5
    def cutout(self, img, size):
        h, w = img.shape[0], img.shape[1]
        x, y = np.random.randint(w), np.random.randint(h)

        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        mask = np.ones_like(img)
        mask[y1:y2, x1:x2] = 0

        return img * mask

    def cutout_set(self, size, p):
        a_images, a_proba, a_types = [], [], []

        for i in range(len(images)):
            if random.random() <= p:
                a_images.append(cutout(images[i], size))
                a_proba.append(proba[i])
                a_types.append(types[i])

        return a_images, a_proba, a_types

