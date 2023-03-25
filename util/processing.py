import cv2
import numpy as np

def linearize_img(img):
    # Convert the image from sRGB to linear RGB
    linear_img = np.power(img / 255.0, 2.2)
    return linear_img

def unlinearize_img(img):
    # Convert the balanced image back to sRGB
    output_img = np.power(img, 1/2.2) * 255.0
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)   
    return output_img

def gw_balance(img):
     gw_balanced_img = cv2.xphoto.balanceWhite(img, None, cv2.xphoto.WHITE_BALANCE_SIMPLE)
     return gw_balanced_img