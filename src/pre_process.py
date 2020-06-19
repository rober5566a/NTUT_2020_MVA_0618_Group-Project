import cv2
import numpy as np
import random
from Model.Img_DIP import *


def change_HSV_V(img, v_value=0):
    img_dip = ImgDIP(img.copy())
    hsv = img_dip.img_hsv

    h, s, v = img_dip.split_channels(hsv)
    v = img_dip.increase_256_channel(v, v_value)

    output_hsv = img_dip.merge_channels(h, s, v)
    output_img = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2BGR)
    output_img = np.reshape(output_img,
                            (1, output_img.shape[0], output_img.shape[1], output_img.shape[2]))

    return output_img


def rotate(img, angle, center=None, scale=1.0):
    img = img.copy()
    (h, w) = img.shape[:2]
    if center is None:
        center = (w/2, h/2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    output_img = cv2.warpAffine(img, M, (w, h))
    output_img = np.reshape(output_img,
                            (1, output_img.shape[0], output_img.shape[1], output_img.shape[2]))

    return output_img


def try_append(imgs, img):
    try:
        imgs = np.append(imgs, img, axis=0)
    except ValueError:
        imgs = img

    return imgs


def dip_pre_process(img, num_create=1):
    imgs = np.array([], dtype='uint8')

    hsv_v_values = []
    rotate_values = []
    for i in range(num_create):
        hsv_v_values.append(random.randint(-100, 100))
        rotate_values.append(random.randint(-180, 180))

    for hsv_v_value in hsv_v_values:
        hsv_img = change_HSV_V(img, v_value=hsv_v_value)
        imgs = try_append(imgs, hsv_img)

    for rotate_value in rotate_values:
        rotate_img = rotate(img, angle=rotate_value)
        imgs = try_append(imgs, rotate_img)

    return imgs


if __name__ == "__main__":
    img = cv2.imread("out/78_train_loss.png")
    rotate(img, 30)
