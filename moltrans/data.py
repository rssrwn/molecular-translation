import torch
import pandas as pd
from PIL import Image
from rdkit import Chem
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor


class BMSDataset(Dataset):
    def __init__(self, img_paths, inchis, transform=None, smiles=False):
        self.img_paths = img_paths
        self.inchis = inchis
        self.transform = transform
        self.smiles = smiles

    @staticmethod
    def from_data_path(data_path, transform=None, smiles=False):
        path = Path(data_path)
        labels_df = pd.read_csv(path / "train_labels.csv")
        img_dict = BMSDataset._load_image_paths(path / "train")

        inchis = labels_df["InChI"].tolist()
        img_paths = [img_dict[img_id] for img_id in labels_df["image_id"].tolist()]

        dataset = BMSDataset(img_paths, inchis, transform=transform, smiles=smiles)
        return dataset

    def __len__(self):
        return len(self.inchis)

    def __getitem__(self, item):
        mol_str = self.inchis[item]
        if self.smiles:
            mol_str = self._convert_to_smiles(mol_str)

        img = self._load_img(self.img_paths[item])
        img = self.transform(img) if self.transform is not None else img

        return img, mol_str

    @staticmethod
    def _load_image_paths(path):
        img_paths = {}
        executor = ProcessPoolExecutor(multiprocessing.cpu_count() * 4)

        for i in path.iterdir():
            for j in i.iterdir():
                futures = [executor.submit(BMSDataset._img_names, k) for k in j.iterdir()]
                results = [future.result() for future in futures]
                for result in results:
                    for name, file in result:
                        img_paths[name] = file

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

    def _convert_to_smiles(self, inchi):
        mol = Chem.inchi.MolFromInchi(inchi)
        smiles = Chem.MolToSmiles(mol)
        return smiles


class BMSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        tokeniser,
        pin_memory=True
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

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
        imgs, mol_strs = tuple(zip(*batch))
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
