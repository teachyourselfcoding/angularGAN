import os
import numpy as np
from PIL import Image

# Function to concatenate images and insert confidence value in the middle
def superimpose_confidence(folder_path, image_prefix):
    fake_B_path = os.path.join(folder_path, f"{image_prefix}_fake_B.png")
    real_A_path = os.path.join(folder_path, f"{image_prefix}_realA.png")
    npy_path = os.path.join(folder_path, f"{image_prefix}.npy")

    fake_B = Image.open(fake_B_path)
    real_A = Image.open(real_A_path)
    concatenated_image = Image.new("RGB", (fake_B.width + real_A.width, fake_B.height))
    concatenated_image.paste(fake_B, (0, 0))
    concatenated_image.paste(real_A, (fake_B.width, 0))

    confidence = np.load(npy_path)
    confidence_str = f"{confidence:.4f}"
    confidence_img = Image.new("RGB", (fake_B.width, fake_B.height), color=(255, 255, 255))
    confidence_draw = ImageDraw.Draw(confidence_img)
    confidence_draw.text((fake_B.width / 2, fake_B.height / 2), confidence_str, fill=(0, 0, 0), align="center")

    final_image = Image.new("RGB", (concatenated_image.width, concatenated_image.height + confidence_img.height))
    final_image.paste(concatenated_image, (0, 0))
    final_image.paste(confidence_img, (0, concatenated_image.height))

    final_image.save(os.path.join(folder_path, f"{image_prefix}_confidence.png"))

# Example usage
folder_path = "/home/yanfeng/results/13march/march12test/test_latest/images"
for filename in os.listdir(folder_path):
    if filename.endswith("_fake_B.png"):
        image_prefix = filename[:-11]
        superimpose_confidence(folder_path, image_prefix)
