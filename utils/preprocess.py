import numpy as np
import cv2
import torch
from PIL import Image


def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def preprocess_image(image: Image.Image, image_size=224):
    image = image.convert("L")
    image = image.resize((image_size, image_size))

    arr = np.array(image)
    arr = apply_clahe(arr)
    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)
    arr = np.repeat(arr, 3, axis=0)

    tensor = torch.tensor(arr, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)

    return tensor