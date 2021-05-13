import math
import pytorch_lightning as pl
from PIL import ImageOps


class Squarify:
    """ Pad shortest dimension of image to be same as longest. """

    def __call__(self, img):
        new_size = (max(img.size), max(img.size))
        padded = ImageOps.pad(img, new_size, color="white")
        return padded


def set_seed(seed):
    pl.utilities.seed.seed_everything(seed)


def calc_train_steps(datam, epochs, acc_batches, gpus=1):
    datam.setup()
    batches_per_gpu = math.ceil(len(datam.train_dataloader()) / float(gpus))
    train_steps = math.ceil(batches_per_gpu / acc_batches) * epochs
    return train_steps
