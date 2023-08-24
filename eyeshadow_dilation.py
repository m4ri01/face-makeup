import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test_2 import evaluate
import argparse
import json

# def parse_args():
#     parse = argparse.ArgumentParser()
#     parse.add_argument("-i",'--img-path', default='imgs/6.jpg')
#     parse.add_argument("-o","--output",default="dilate_segmented.png")
#     return parse.parse_args()


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


def change_color(image, parsing, part=17, color=[255,255,255]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r
    
    tar_color[parsing != part] = image[parsing != part]
    return tar_color

def change_color_inverse(image, parsing, part=17, color=[255,255,255]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r
    
    tar_color[parsing == part] = image[parsing == part]
    return tar_color

def change_color_gray(image,parsing,parts):
    b1,g1,r1 = [255,255,255]
    tar_color1 = np.zeros_like(image)
    tar_color1[:,:,0] = b1
    tar_color1[:,:,1] = g1
    tar_color1[:,:,2] = r1
    
    tar_color2 = np.zeros_like(image)
    parts = np.array(parts)
    tar_color2[np.isin(parsing,parts)] = tar_color1[np.isin(parsing,parts)]
    return tar_color2

def dilate_image(image,kernel_size=(5,5),iterate=1):
    dilation = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones(kernel_size,np.uint8)
    dilation = cv2.dilate(dilation,kernel,iterations=iterate)
    # for i in range(20):
    #     dilation = cv2.dilate(dilation,kernel,iterations=iterate)
    # # dilation = cv2.dilate(image_gray,kernel,iterations=iterate)
    # # dilation = cv2.dilate(dilation,kernel,iterations=iterate)
    return dilation
    
def change_eyeshadow(dilation_image,parsing,parts):
    # grayscale_image = cv2.cvtColor(grayscale_image,cv2.COLOR_BGR2GRAY)
    tar_color = np.zeros_like(dilation_image)
    
    dilation_image[np.isin(parsing,parts)] = tar_color[np.isin(parsing,parts)]
    return dilation_image

def get_coordinate_wh(parsing,parts):
    dict_result = {}
    for p in parts:
        coordinat_orig = []
        for x in range(parsing.shape[0]):
            for y in range(parsing.shape[1]):
                if parsing[x,y] == p:
                    coordinat_orig.append([y,x])
        coordinat = np.array(coordinat_orig)
        w = int(np.max(coordinat[:,0]) - np.min(coordinat[:,0]))
        h = int(np.max(coordinat[:,1]) - np.min(coordinat[:,1]))
        if p == 4:
            dict_result["l"] = {"coordinat":coordinat_orig,"w":w,"h":h}
        else:
            dict_result["r"] = {"coordinat":coordinat_orig,"w":w,"h":h}
    return dict_result


if __name__ == '__main__':
    # 1  face
    # 11 teeth
    # 12 upper lip
    # 13 lower lip
    # 17 hair

    # args = parse_args()

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
    
    src_dir = "C:\\Users\\GULO\\Documents\\KEISUU\\JT-Dataset\\images\\non-makeup"
    dest_dir = "C:\\Users\\GULO\\Documents\\KEISUU\\JT-Dataset\\images\\result_non_makeup"
    
    files = [file for file in os.listdir(src_dir)]
    
    for f in files:
        print(f)
        try:
            # image_path = args.img_path
            image_path = os.path.join(src_dir,f)
            cp = 'cp/79999_iter.pth'
        
            image = cv2.imread(image_path)
            ori = image.copy()
            parsing = evaluate(image_path, cp)
            parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
            
            parts = [table['eyes_l'],table['eyes_r']]
            colors = [[255,255,255],[255,255,255]]
            # get_cwh = get_coordinate_wh(parsing, parts)
            grayscale_image = change_color_gray(ori,parsing,parts)   
            dilation_image = dilate_image(grayscale_image,kernel_size=(10,10),iterate=20)
            eyeshadow_image = change_eyeshadow(dilation_image,parsing,parts)
            
            image_segmented_remove = change_color(ori,eyeshadow_image,255)
            image_eyes = change_color_inverse(ori, eyeshadow_image,255)
        
            cv2.imwrite("{}".format(os.path.join(dest_dir,f)),image_segmented_remove)
            out_inverse = "inverse_{}".format(f)
            cv2.imwrite("{}".format(os.path.join(dest_dir,out_inverse)),image_eyes)
        except:
            print("Error!")
    # a = json.dumps(get_cwh)
    # with open("coordinat_wh.json",'w') as f:
    #     json.dump(get_cwh,f)












