import os
import torch
import csv
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from .naive_dataset import NaiveDataset, SimpleFewShotSampler

torch.manual_seed(42)
np.random.seed(42)


fs_label_splits = [
    [8, 5, 6, 11, 15, 12, 20, 26, 24, 35, 33, 38, 40, 48, 47],
    [5, 0, 4, 18, 13, 19, 29, 21, 28, 33, 38, 31, 47, 46, 44],
    [0, 9, 7, 18, 13, 17, 27, 24, 25, 38, 31, 35, 42, 47, 46],
    [5, 7, 9, 18, 19, 12, 23, 28, 27, 36, 30, 32, 46, 41, 44],
    [3, 8, 5, 10, 17, 14, 29, 27, 24, 38, 35, 32, 47, 42, 46]
]

class ESC50(NaiveDataset):
    r"""ESC-50 dataset."""
    def __init__(
            self,
            audio_dir: str,
            csv_path: str, *,
            fold: list = [1, 2, 3, 4, 5],  # do NOT change in the few-shot setting
            data_type: str = 'path',
            sample_rate: int = 44100,
            target_type: str = 'category',
            **kargs
    ) -> None:
        super().__init__()
        self.audio_dir = audio_dir
        self.meta = self._load_meta(csv_path, fold)  # {filename: child_id}
        assert len(self.meta) > 0
        self.indices = list(self.meta.keys())  # create indices for filename
        self.cfgs = {
            'data_type': data_type,
            'target_type': target_type
        }
        if self.cfgs['data_type'] == 'audio':
            self.cfgs['sr'] = sample_rate
        if self.cfgs['target_type'] == 'category':
            self.detokeniser = self._create_detokeniser(csv_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: Tuple[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns tensor of audio and its label."""
        x, y = item
        x = os.path.join(self.audio_dir, x)
        if self.cfgs['data_type'] == 'audio':
            x = self._load_audio(x, sr=self.cfgs.sr)
        if self.cfgs['target_type'] == 'category':
            y = self.detokeniser[y]
        return x, y

    def _load_meta(self, csv_path: str, fold: list) -> dict:
        r"""Load meta information.
        Args:
            csv_path: str, path to `esc50.csv`
            fold: list, fold id(s) needed for train/val dataset
        Returns:
            Dict of format: filename -> class_id 
        """
        meta = dict()
        with open(csv_path, 'r') as f:  # esc50.csv organise the meta in the terms of
            rows = csv.DictReader(f)  # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
            for r in rows:
                if int(r['fold']) in fold:
                    meta[r['filename']] = int(r['target'])  # convert to int idx from str type from csv file
        return meta

    def _create_detokeniser(self, csv_path: str) -> dict:
        r"""Returns a detokeniser of format: class_id -> class_name."""
        detokeniser = dict()
        with open(csv_path, 'r') as f:  # esc50.csv organise the meta in the terms of
            rows = csv.DictReader(f)  # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
            for r in rows:
                t = int(r['target'])
                if t not in detokeniser.keys():
                    detokeniser[t] = r['category']
                else:
                    assert detokeniser[t] == r['category']
        return detokeniser

    @property
    def labelset(self):
        if self.cfgs['target_type'] == 'category':
            return list(set(list(self.detokeniser.values())))
        else:
            return list(set(list(self.meta.values())))
            

if __name__ == '__main__':
    audio_dir = '/data/EECS-MachineListeningLab/datasets/ESC-50/audio'
    csv_path = '/data/EECS-MachineListeningLab/datasets/ESC-50/meta/esc50.csv'
    esc50 = ESC50(audio_dir=audio_dir, csv_path=csv_path, data_type='path', target_type='category')
    val_labelset = fs_label_splits[1]
    base_labelset = [x for x in range(len(esc50.labelset)) if x not in val_labelset]
    sampler = SimpleFewShotSampler(dataset=esc50, labelset=val_labelset, n_class=15, n_supports=1, n_queries=5, n_task=100)
    dataloader = DataLoader(esc50, batch_sampler=sampler, num_workers=4, pin_memory=True)
    print(len(dataloader))
    for x, y in dataloader:
        print(f"x={x}\n")
        print(f"target={y}\n")
