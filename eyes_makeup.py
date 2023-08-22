import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test_2 import evaluate
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-i','--img-path', default='imgs/6.jpg')
    parse.add_argument("-o","--output",default="segmented.png")
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def change_color(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    tar_color[parsing != part] = image[parsing != part]
    return tar_color

def change_color_inverse(image,parsing,parts,color=[255,255,255]):
    b,g,r = color
    tar_color = np.zeros_like(image)
    tar_color[:,:,0] = b
    tar_color[:,:,1] = g
    tar_color[:,:,2] = r
    parts = np.array(parts)    
    tar_color[np.isin(parsing,parts)] = image[np.isin(parsing,parts)]
    return tar_color

if __name__ == '__main__':
    # 1  face
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair

    args = parse_args()

    table = {
        # 'face': 1,
        # 'eyebrows_l': 2,
        # 'eyebrows_r': 3,
        'eyes_l':4,
        'eyes_r':5
        # 'ears': 8,
        # 'nose':10,
        # 'teeth': 11,
        # 'hair': 17,
        # 'upper_lip': 12,
        # 'lower_lip': 13
    }

    image_path = args.img_path
    cp = 'cp/79999_iter.pth'

    image = cv2.imread(image_path)
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
   
    parts = [table['eyes_l'],table['eyes_r']]
    colors = [[255,255,255],[255,255,255]]

    for part, color in zip(parts, colors):
        image = change_color(image, parsing, part, color)
    
    image_eyes = change_color_inverse(ori, parsing, parts)
    
    cv2.imwrite("orig.png",ori)
    cv2.imwrite(args.output,image)
    cv2.imwrite("inverse_{}".format(args.output),image_eyes)















