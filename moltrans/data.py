import os
import torch
import multiprocessing
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from rdkit import Chem
from pathlib import Path
from functools import partial
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor


class BMSDataset(Dataset):
    def __init__(self, img_paths, inchis, transform=None):
        self.img_paths = img_paths
        self.inchis = inchis
        self.transform = transform

    @staticmethod
    def from_data_path(data_path, transform=None):
        path = Path(data_path)
        labels_df = pd.read_csv(path / "train_labels.csv")
        img_dict = BMSDataset._load_image_paths(path / "train")

        inchis = labels_df["InChI"].tolist()
        img_paths = [img_dict[img_id] for img_id in labels_df["image_id"].tolist()]

        dataset = BMSDataset(img_paths, inchis, transform=transform)
        return dataset

    def __len__(self):
        return len(self.inchis)

    def __getitem__(self, item):
        inchi = self.inchis[item]
        mol = Chem.inchi.MolFromInchi(inchi)

        img = self._load_img(self.img_paths[item])
        img = self.transform(img) if self.transform is not None else img

        return img, mol

    @staticmethod
    def _load_image_paths(path):
        img_paths = {}
        executor = ProcessPoolExecutor(multiprocessing.cpu_count() * 4)

        for i in path.iterdir():
            for j in i.iterdir():
                futures = [executor.submit(BMSDataset._img_names, k) for k in j.iterdir()]
                results = [future.result() for future in futures]
                for result in results:
                    for img_name, img_file in result:
                        img_paths[img_name] = img_file

        return img_paths

    @staticmethod
    def _img_names(path):
        img_names = []
        for img_file in path.iterdir():
            img_name = str(img_file)[0:-4].split("/")[-1]
            img_names.append((img_name, img_file))

        return img_names

    def _load_img(self, path):
        img = Image.open(path)
        return img


class BMSTestDataset(BMSDataset):
    def __init__(self, img_ids, img_paths, transform=None):
        self.img_ids = img_ids
        self.img_paths = img_paths
        self.transform = transform

    @staticmethod
    def from_data_path(data_path, transform=None):
        path = Path(data_path)
        img_dict = BMSDataset._load_image_paths(path / "test")
        img_ids = list(img_dict.keys())
        img_paths = [img_dict[img_id] for img_id in img_ids]
        dataset = BMSTestDataset(img_ids, img_paths, transform=transform)
        return dataset

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, item):
        img_id = self.img_ids[item]
        img = self._load_img(self.img_paths[item])
        img = self.transform(img) if self.transform is not None else img
        return img_id, img


class BMSExtraMolDataset(Dataset):
    def __init__(self, inchis):
        self.inchis = inchis

    @staticmethod
    def from_data_path(data_path):
        csv_path = Path(data_path) / "extra_approved_InChIs.csv"
        df = pd.read_csv(csv_path)
        inchis = df["InChI"].tolist()
        dataset = BMSExtraMolDataset(inchis)
        return dataset

    def __len__(self):
        return len(self.inchis)

    def __getitem__(self, item):
        inchi = self.inchis[item]
        mol = Chem.inchi.MolFromInchi(inchi)
        return mol


# ******************************************************************************************
# ************************************** Data Modules **************************************
# ******************************************************************************************


class BMSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        tokeniser,
        pin_memory=True,
        aug_mols=True
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.tokeniser = tokeniser
        self.pin_memory = pin_memory
        self.aug_mols = aug_mols

        self._num_workers = len(os.sched_getaffinity(0))

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=self._collate,
            shuffle=True,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=partial(self._collate, train=False),
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=partial(self._collate, train=False),
            pin_memory=self.pin_memory
        )
        return loader

    def _collate(self, batch, train=True):
        imgs, mols = tuple(zip(*batch))
        if self.aug_mols and train:
            mols = [self._augment_mol(mol) for mol in mols]

        mol_strs = [Chem.MolToSmiles(mol, canonical=False) for mol in mols]
        token_output = self.tokeniser.tokenise(mol_strs, pad=True)
        tokens = token_output["original_tokens"]
        pad_masks = token_output["original_pad_masks"]

        imgs_batch = torch.stack(imgs)
        token_ids = self.tokeniser.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor(token_ids).transpose(0, 1)
        pad_masks = torch.tensor(pad_masks, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            "images": imgs_batch,
            "decoder_input": token_ids[:-1, :],
            "decoder_pad_mask": pad_masks[:-1, :],
            "target": token_ids.clone()[1:, :],
            "target_mask": pad_masks.clone()[1:, :],
            "target_string": mol_strs
        }
        return collate_output

    def _augment_mol(self, mol):
        atom_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_order)
        aug_mol = Chem.RenumberAtoms(mol, atom_order)
        return aug_mol


class BMSImgDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, tokeniser, transform, pin_memory=True):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokeniser = tokeniser
        self.transform = transform
        self.pin_memory = pin_memory
        self._num_workers = len(os.sched_getaffinity(0))

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=self._collate,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=partial(self._collate, train=False),
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return loader

    def _collate(self, batch, train=True):
        _, imgs = tuple(zip(*batch))
        imgs1 = [self.transform(img) for img in imgs]
        imgs2 = [self.transform(img) for img in imgs]
        imgs1 = torch.stack(imgs1)
        imgs2 = torch.stack(imgs2)
        return (imgs1, imgs2)


class BMSSmilesDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, tokeniser, pin_memory=True):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.tokeniser = tokeniser
        self.pin_memory = pin_memory
        self._num_workers = len(os.sched_getaffinity(0))

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=self._collate,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=partial(self._collate, train=False),
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return loader

    def _collate(self, batch, train=True):
        # Create encoder input
        mols = [self._augment_mol(mol) for mol in batch]
        mol_strs = [Chem.MolToSmiles(mol, canonical=False) for mol in mols]

        token_output = self.tokeniser.tokenise(mol_strs, pad=True, mask=True)
        enc_tokens = token_output["masked_tokens"]
        enc_pad_masks = token_output["masked_pad_masks"]

        enc_token_ids = self.tokeniser.convert_tokens_to_ids(enc_tokens)
        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
        enc_pad_masks = torch.tensor(enc_pad_masks, dtype=torch.bool).transpose(0, 1)

        # Create decoder input
        aug_mols = [self._augment_mol(mol) for mol in mols]
        aug_mol_strs = [Chem.MolToSmiles(mol, canonical=False) for mol in aug_mols]

        token_output = self.tokeniser.tokenise(aug_mol_strs, pad=True)
        dec_tokens = token_output["original_tokens"]
        dec_pad_masks = token_output["original_pad_masks"]

        dec_token_ids = self.tokeniser.convert_tokens_to_ids(dec_tokens)
        dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
        dec_pad_masks = torch.tensor(dec_pad_masks, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_masks,
            "decoder_input": dec_token_ids[:-1, :],
            "decoder_pad_mask": dec_pad_masks[:-1, :],
            "target": dec_token_ids.clone()[1:, :],
            "target_mask": dec_pad_masks.clone()[1:, :]
        }
        return collate_output

    def _augment_mol(self, mol):
        atom_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_order)
        aug_mol = Chem.RenumberAtoms(mol, atom_order)
        return aug_mol
