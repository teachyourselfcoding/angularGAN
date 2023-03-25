import cv2
import numpy as np
from processing import linearize_img
from processing import gw_balance
from processing import unlinearize_img
img_path = 
img = cv2.imread(img_path)

# linearize image
linear_img = linearize_img(img)
output_img = np.concatenate((img, linear_img), axis=1)
# Display the original and corrected images
cv2.imshow('Original and Gamma Correction', output_img)

gw_balanced_img = gw_balance(linear_img)
gw_balanced_img2 = gw_balance(img)
output_img2 = np.concatenate((gw_balanced_img2, gw_balanced_img), axis=1)
cv2.imshow('After correction', output_img)

unlinarized_img = unlinearize_img(gw_balanced_img)
unlinarized_img2 = unlinearize_img(gw_balanced_img2)
output_img3 = np.concatenate((unlinarized_img2, unlinarized_img), axis=1)
cv2.imshow('After unlinarize', output_img3)