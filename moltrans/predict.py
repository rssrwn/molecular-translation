import os
import argparse
import pandas as pd
from rdkit import Chem
from pathlib import Path
from torch.utils.data import DataLoader

import moltrans.util as util
from moltrans.data import BMSTestDataset
from moltrans.model import BMSEncoder, BMSDecoder, BMSModel


DEFAULT_VOCAB_PATH = "vocab.txt"
DEFAULT_MAX_SEQ_LEN = 256

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
    encoder = BMSEncoder(D_MODEL)
    decoder = BMSDecoder(D_MODEL, D_FEEDFORWARD, NUM_LAYERS, NUM_HEADS)
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
        model_input = {"images": torch.stack(imgs)}
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
    test_dataset = BMSTestDataset.from_data_path(args.data_path, transform=util.TRANSFORM)
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

    parsed_args = parser.parse_args()
    main(parsed_args)
