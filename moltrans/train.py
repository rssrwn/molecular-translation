import math
import random
import argparse
import pytorch_lightning as pl
import torchvision.transforms as T
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from moltrans.data import BMSDataset, BMSDataModule


RANDOM_SEED = 42069
VAL_SPLIT = 0.05
CHEM_TOKEN_START = 26
LOG_DIR = "tb_logs"
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

# Defaults
DEFAULT_VOCAB_PATH = "vocab.txt"
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 1
DEFAUlt_ACC_BATCHES = 1
DEFAULT_CLIP_GRAD = 1.0
DEFAULT_D_MODEL = 512
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 6
DEFAULT_SCHEDULE = "cycle"
DEFAULT_LR = 0.001

IMG_SIZE = (256, 256)
IMG_MEAN = 0.9871
IMG_STD_DEV = 0.08968

TRANSFORM = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(IMG_MEAN, IMG_STD_DEV)
])


def set_seed():
    pl.utilities.seed.seed_everything(RANDOM_SEED)


def calc_train_steps(datam, epochs, acc_batches, gpus=1):
    datam.setup()
    batches_per_gpu = math.ceil(len(datam.train_dataloader()) / float(gpus))
    train_steps = math.ceil(batches_per_gpu / acc_batches) * epochs
    return train_steps


def split_dataset(dataset, split, train_transform=None, val_transform=None, smiles=False):
    num_val = round(split * len(dataset))

    val_idxs = random.sample(range(len(dataset)), num_val)
    val_img_paths = [dataset.img_paths[idx] for idx in val_idxs]
    val_inchis = [dataset.inchis[idx] for idx in val_idxs]
    val_dataset = BMSDataset(val_img_paths, val_inchis, transform=val_transform, smiles=smiles)

    train_idxs = list(set(range(len(dataset))) - set(val_idxs))
    train_img_paths = [dataset.img_paths[idx] for idx in train_idxs]
    train_inchis = [dataset.inchis[idx] for idx in train_idxs]
    train_dataset = BMSDataset(train_img_paths, train_inchis, transform=train_transform, smiles=smiles)

    return train_dataset, val_dataset


def build_model(args, dm, sampler, vocab_size):
    train_steps = calc_train_steps(dm, args.epochs, args.acc_batches)
    encoder = BMSEncoder(args.d_model)
    decoder = BMSDecoder(args.d_model, args.d_feedforward, args.num_layers, args.num_heads)
    model = BMSModel(
        encoder,
        decoder,
        args.d_model,
        sampler,
        args.lr,
        vocab_size,
        args.max_seq_len,
        args.schedule,
        train_steps
    )
    return model


def build_trainer(args):
    precision = 16 if torch.cuda.is_available() else 32
    gpus = 1 if torch.cuda.is_available() else 0
    logger = TensorBoardLogger(LOG_DIR, name="bms-mol")
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
    # Load dataset
    dataset = BMSDataset.from_data_path(args.data_path)

    # Build tokeniser and sampler
    tokeniser = Tokeniser.from_vocab_file(args.vocab_path, REGEX, CHEM_TOKEN_START)
    sampler = DecodeSampler(tokeniser, args.max_seq_len)

    # Split dataset randomly
    train_dataset, val_dataset = split_dataset(
        dataset,
        VAL_SPLIT,
        train_transform=TRANSFORM,
        val_transform=TRANSFORM,
        smiles=True
    )

    # Build data module
    dm = BMSDataModule(train_dataset, val_dataset, None, args.batch_size, tokeniser)

    # Build model
    vocab_size = len(tokeniser)
    model = build_model(args, dm, sampler, vocab_size)

    # Build PL trainer
    trainer = build_trainer(args)

    # Fit training data
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
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

    parsed_args = parser.parse_args()
    main(parsed_args)
