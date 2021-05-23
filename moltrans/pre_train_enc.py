import math
import torch
import random
import argparse
import pytorch_lightning as pl
import torchvision.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import moltrans.util as util
from moltrans.tokeniser import Tokeniser
from moltrans.sampler import DecodeSampler
from moltrans.data import BMSTestDataset, BMSImgDataModule
from moltrans.model import MoCoEncoder


RANDOM_SEED = 42069
VAL_SPLIT = 0.02
CHEM_TOKEN_START = 26
LOG_DIR = "tb_logs"
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

# Defaults
DEFAULT_VOCAB_PATH = "vocab.txt"
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 10
DEFAULT_ACC_BATCHES = 1
DEFAULT_CLIP_GRAD = 1.0
DEFAULT_D_MODEL = 512
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 6
DEFAULT_SCHEDULE = "cycle"
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_WARM_UP_STEPS = 4000


def split_dataset(dataset, split, train_transform=None, val_transform=None):
    num_val = round(split * len(dataset))

    val_idxs = random.sample(range(len(dataset)), num_val)
    val_img_ids = [dataset.img_ids[idx] for idx in val_idxs]
    val_img_paths = [dataset.img_paths[idx] for idx in val_idxs]
    val_dataset = BMSTestDataset(val_img_ids, val_img_paths, transform=val_transform)

    train_idxs = list(set(range(len(dataset))) - set(val_idxs))
    train_img_ids = [dataset.img_ids[idx] for idx in train_idxs]
    train_img_paths = [dataset.img_paths[idx] for idx in train_idxs]
    train_dataset = BMSTestDataset(train_img_ids, train_img_paths, transform=train_transform)

    return train_dataset, val_dataset


def build_model(args, dm, sampler, vocab_size):
    extra_args = {
        "acc_batches": args.acc_batches
    }
    train_steps = util.calc_train_steps(dm, args.epochs, args.acc_batches)
    model = MoCoEncoder(
        args.d_model,
        args.d_feedforward,
        args.num_layers,
        args.num_heads,
        args.lr,
        args.max_seq_len,
        args.schedule,
        train_steps,
        args.weight_decay,
        args.warm_up_steps,
        **extra_args
    )
    return model


def build_trainer(args):
    precision = 16 if torch.cuda.is_available() else 32
    gpus = 1 if torch.cuda.is_available() else 0
    logger = TensorBoardLogger(LOG_DIR, name="bms-encoder")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="val_lev_dist", save_last=True)
    callbacks = [lr_monitor, checkpoint_cb]

    print(f"Using precision: {precision}-bit")

    trainer = pl.Trainer(
        logger=logger,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.clip_grad,
        callbacks=callbacks,
        precision=precision,
        gpus=gpus
    )
    return trainer


def main(args):
    util.set_seed(RANDOM_SEED)

    # Load dataset
    print("Loading dataset...")
    dataset = BMSTestDataset.from_data_path(args.data_path)
    print("Dataset complete.")

    # Build tokeniser and sampler
    print("Loading tokeniser and sampler...")
    tokeniser = Tokeniser.from_vocab_file(args.vocab_path, REGEX, CHEM_TOKEN_START)
    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    print("Complete.")

    # Split dataset randomly
    print("Spliting dataset...")
    train_dataset, val_dataset = split_dataset(dataset, VAL_SPLIT)
    print("Complete.")

    # Build data module
    print("Loading data module...")
    dm = BMSImgDataModule(train_dataset, val_dataset, args.batch_size, tokeniser, util.PRE_TRAIN_ENC_TRANSFORM)
    print("Data module complete.")

    # Build model
    print("Building model...")
    vocab_size = len(tokeniser)
    model = build_model(args, dm, sampler, vocab_size)
    print("Complete.")

    # Build PL trainer
    print("Building trainer...")
    trainer = build_trainer(args)
    print("Complete.")

    # Fit training data
    print("Fitting data module to model...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--vocab_path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)

    # Model and training args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_CLIP_GRAD)
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--d_feedforward", type=int, default=DEFAULT_D_FEEDFORWARD)
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)

    parsed_args = parser.parse_args()
    main(parsed_args)
