import cv2
import numpy as np
from albumentations.augmentations.geometric.transforms import *
import albumentations

tfs = albumentations.Compose([Affine(rotate=[-7, 7], shear=None, p=0.5),
                                  Perspective(scale=(0.02, 0.1), p=0.1)])

def transform_plate(plate, tfs=tfs):
    
    plate = np.array(cv2.cvtColor(plate, cv2.COLOR_RGB2HSV), dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    plate[:, :, 2] = plate[:, :, 2] * random_bright
    plate[:, :, 2][plate[:, :, 2] > 255] = 255
    plate = cv2.cvtColor(np.array(plate, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    
    return tfs(image=plate)["image"]
