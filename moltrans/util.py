import math
import torch
import random
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import ImageOps


class Squarify:
    """ Pad shortest dimension of image to be same as longest. Only for PIL images. """

    def __call__(self, img):
        new_size = (max(img.size), max(img.size))
        padded = ImageOps.pad(img, new_size, color="white")
        return padded


class BlurExtraChannels:
    """ Create two additional image channels by blurring. Only for Tensors """

    def __init__(self, sigma_1, simga_2):
        kernel_size = 7
        self.blur_1 = T.GaussianBlur(kernel_size, sigma=sigma_1)
        self.blur_2 = T.GaussianBlur(kernel_size, sigma=sigma_2)

    def __call__(self, img):
        blur_1 = self.blur_1(img)
        blur_2 = self.blur_2(img)
        concat_img = torch.stack((img, blur_1, blur_2)).transpose(0, 1).unsqueeze(2)
        return concat_img


class RandomPad:
    """ Randomly pad each side of the image. Only for PIL images. """

    def __init__(self, p, pad_range):
        self.p = p
        self.pad_range = pad_range

    def __call__(self, img):
        if random.random() < self.p:
            border = random.randint(*self.pad_range)
            img = ImageOps.expand(img, border=border, fill="white")

        return img


USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()

IMG_SIZE = (256, 256)
IMG_MEAN = 0.9871
IMG_STD_DEV = 0.08968
WHITE = 255

PAD_PROB = 0.5
PAD_RANGE = (10, 50)
ROTATION_RANGE = (-180, 180)

TRAIN_TRANSFORM = T.Compose([
    Squarify(),
    T.Resize(IMG_SIZE),
    # RandomPad(PAD_PROB, PAD_RANGE),
    # T.Resize(IMG_SIZE),
    # T.RandomRotation(ROTATION_RANGE, fill=WHITE),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD_DEV)
])

TEST_TRANSFORM = T.Compose([
    Squarify(),
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD_DEV)
])


def set_seed(seed):
    pl.utilities.seed.seed_everything(seed)


def calc_train_steps(datam, epochs, acc_batches, gpus=1):
    datam.setup()
    batches_per_gpu = math.ceil(len(datam.train_dataloader()) / float(gpus))
    train_steps = math.ceil(batches_per_gpu / acc_batches) * epochs
    return train_steps
