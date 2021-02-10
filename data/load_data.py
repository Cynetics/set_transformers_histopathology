import torchvision.transforms as transforms
from PIL import Image
from .barrett_loader import *
import pathlib

def get_barrett_data(config=None):
    print(pathlib.Path().absolute())

    training_data = BarrettData(img_dir="./data/WSI_patches/train/")
    return training_data

def get_barrett_val_data(config=None):
    print(pathlib.Path().absolute())
    val_data = BarrettPatchValData(img_dir="./data/WSI_patches/val/")
    return val_data