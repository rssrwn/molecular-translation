import os
import argparse
import torch
import pandas as pd
from rdkit import Chem
from pathlib import Path
from torch.utils.data import DataLoader

import moltrans.util as util
from moltrans.data import BMSTestDataset
from moltrans.tokeniser import Tokeniser
from moltrans.sampler import DecodeSampler
from moltrans.model import MoCoEncoder, BARTModel, BMSModel


RANDOM_SEED = 42069
VAL_SPLIT = 0.02
CHEM_TOKEN_START = 26
LOG_DIR = "tb_logs"
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

DEFAULT_VOCAB_PATH = "vocab.txt"
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_BATCH_SIZE = 128
DEFAULT_ENC_PATH = "tb_logs/bms-encoder/version_1/checkpoints/last.ckpt"
DEFAULT_DEC_PATH = "tb_logs/bms-decoder/version_0/checkpoints/last.ckpt"

D_MODEL = 512
D_FEEDFORWARD = 2048
NUM_HEADS = 8
NUM_LAYERS = 6


def build_dataloader(args, dataset):
    num_workers = len(os.sched_getaffinity(0))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    return loader


def load_model(args, sampler):
    # Load encoder
    encoder = MoCoEncoder.load_from_checkpoint(args.encoder_path)
    encoder.encoder_k = None
    encoder.enc_k_fc = None
    encoder.queue = None
    encoder.queue_ptr = None

    # Load decoder
    decoder = BARTModel.load_from_checkpoint(args.decoder_path)
    decoder.encoder = None

    model = BMSModel.load_from_checkpoint(
        args.model_path,
        encoder=encoder,
        decoder=decoder,
        sampler=sampler,
        max_seq_len=args.max_seq_len
    )
    model.eval()
    return model


def predict(model, loader):
    device = "cuda:0" if util.use_gpu else "cpu"
    model = model.to(device)
    model.eval()

    num_batches = len(loader)

    img_ids = []
    sampled_smiles = []
    for idx, (batch_img_ids, imgs) in enumerate(loader):
        model_input = {"images": imgs.to(device)}
        mol_strs, _ = model.sample_molecules(model_input)
        img_ids.extend(batch_img_ids)
        sampled_smiles.extend(mol_strs)
        print(f"Completed batch: {str(idx)}/{str(num_batches)}")

    example_inchi = "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"
    sampled_mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
    invalid = [mol is None for mol in sampled_mols]
    sampled_inchis = [Chem.inchi.MolToInchi(mol) if mol is not None else example_inchi for mol in sampled_mols]
    return img_ids, sampled_inchis


def write_predictions(filename, img_ids, inchis):
    df_data = {
        "image_id": img_ids,
        "InChI": inchis
    }
    df = pd.DataFrame(data=df_data)
    df.to_csv(Path(filename))


def main(args):
    print("Loading test dataset...")
    test_dataset = BMSTestDataset.from_data_path(args.data_path, transform=util.TEST_TRANSFORM)
    print("Loaded dataset.")

    print("Loading tokeniser and sampler...")
    tokeniser = Tokeniser.from_vocab_file(args.vocab_path, REGEX, CHEM_TOKEN_START)
    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    print("Complete.")

    print("Building dataloader...")
    test_loader = build_dataloader(args, test_dataset)
    print("Built dataloader.")

    print("Loading model...")
    model = load_model(args, sampler)
    print("Loaded model.")

    print("Generating predictions...")
    img_ids, inchis = predict(model, test_loader)
    print("Complete.")

    print("Writing predictions...")
    write_predictions(args.output_file, img_ids, inchis)
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--vocab_path", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--encoder_path", type=str, default=DEFAULT_ENC_PATH)
    parser.add_argument("--decoder_path", type=str, default=DEFAULT_DEC_PATH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

    parsed_args = parser.parse_args()
    main(parsed_args)
