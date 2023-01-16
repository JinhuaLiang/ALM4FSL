import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, List
from .naive_dataset import NaiveDataset, SimpleFewShotSampler

torch.manual_seed(42)
np.random.seed(42)


fsd50k_select_ids = {
    'base': [
            '/m/07rgt08', '/m/0ytgt', '/m/02zsn', '/m/07sr1lc', '/m/07s0dtb', '/m/07r660_', '/m/01h8n0', '/m/05zppz',
            '/m/01j3sz', '/m/03qc9zr', '/m/07pbtc8', '/m/025_jnm', '/m/053hz1', '/m/01b_21', '/m/03cczk', '/m/03qtwd',
            '/m/07rkbfh', '/m/0l15bq', '/m/02rtxlg', '/m/01hsr_', '/m/015lz1', '/m/03q5_w', '/m/07plz5l', '/m/028ght',
            '/m/02yds9', '/m/01dwxx', '/m/07qrkrw', '/m/020bb7', '/m/05tny_', '/m/04s8yn', '/m/025rv6n', '/m/09ld4',
            '/m/01yrx', '/m/0bt9lr', '/m/05r5c', '/m/01qbl', '/m/0j45pbj', '/m/026t6', '/m/07gql', '/m/0342h',
            '/m/07brj', '/m/0mbct', '/m/0l14_3', '/m/0l14md', '/m/05148p4', '/m/01hgjl', '/m/03qjg', '/m/0mkg',
            '/m/03m5k', '/m/07r10fb', '/m/06mb1', '/m/05kq4', '/m/0j6m2', '/m/0btp2', '/m/04_sv', '/m/07r04',
            '/m/01m2v', '/m/0c2wf', '/m/0k4j', '/m/01bjv', '/m/06_fw', '/m/07jdr', '/m/01hnzm', '/m/0199g',
            '/m/01d380', '/m/02y_763', '/m/07rjzl8', '/m/0fqfqc', '/m/07prgkl', '/m/0dxrf', '/m/0642b4', '/m/01x3z',
            '/m/07p7b8y', '/m/07rn7sz', '/m/081rb', '/m/01s0vc', '/m/0dv3j', '/m/01b82r', '/m/02bm9n', '/m/0_ksk',
            '/m/03dnzn', '/m/03l9g', '/m/019jd', '/m/07q7njn', '/m/03kmc9', '/m/0fx9l', '/m/07pb8fc', '/m/02dgv',
            '/m/07q2z82', '/m/05mxj0q', '/m/0g6b5', '/m/0dv5r', '/m/06d_3', '/m/032s66', '/m/01lsmm', '/m/07q8k13',
            '/t/dd00112', '/m/07qcx4z'
        ],
        'val': [
            '/t/dd00003', '/m/0brhx', '/m/09x0r', '/m/02_nn', '/m/0lyf6', '/m/0463cq4', '/m/09b5t', '/m/0ghcn6',
            '/m/03vt0', '/m/02hnl', '/m/05r5wn', '/m/085jw', '/m/0fx80y', '/m/034srq', '/m/07swgks',
            '/m/07pqc89', '/m/0195fx', '/m/0cmf2', '/m/07cx4', '/t/dd00130', '/m/07rrlb6', '/m/01m4t',
            '/m/03v3yw', '/m/0130jx', '/m/02x984l', '/m/023pjk', '/m/02jz0l', '/m/0k5j', '/m/04brg2',
            '/m/07plct2'
        ],
        'eval': [
            '/t/dd00004', '/m/07p6fty', '/m/06h7j', '/m/09xqv', '/m/015p6', '/m/013y1f', '/m/01kcd', '/m/0ngt1',
            '/m/07qjznl', '/m/0316dw', '/m/01jt3m', '/m/0242l', '/m/07qqyl4', '/m/012f08', '/m/07r5v4s'
        ]
}


class FSD_FS(NaiveDataset):
    r"""FSD-FS dataset."""
    clip_cfgs = {
        'sample_rate': 44100,
        'duration': 1,
        'hop_length': 0.5
    }
    def __init__(
            self,
            clip_dir: str,
            audio_dir: str,
            csv_path: str, *,
            mode: str = 'base',
            data_type: str = 'path',
            target_type: str = 'category'
    ) -> None:
        super().__init__()
        self.cfgs = {
            'data_type': data_type,
            'target_type': target_type
        }
        self.clip_dir, self.csv_dir, self.audio_dir = clip_dir, csv_path, os.path.join(audio_dir, mode)
        # Prepare clip-level meta info
        self.meta = dict()
        os.makedirs(self.clip_dir, exist_ok=True)
        if len(os.listdir(self.clip_dir)) == 0:
            print("Start to curate audio clips.")
            csv_path = os.path.join(self.csv_dir, f"{mode}.csv")
            seg_meta = self._read_csv(csv_path)
            clip_data = self._make_clips(seg_meta)
            self._batch_dump(clip_data['clip_audio'], clip_data['clip_name'])
            # Get meta info and dump to csv file
            _str_cats, _fmt = list(), ','
            for name, cat in zip(clip_data['clip_name'], clip_data['clip_category']):
                self.meta[name] = cat
                _str_cats.append(_fmt.join(cat))
            pd.DataFrame(
                {'file_name': clip_data['clip_name'], 'category': _str_cats}
                ).to_csv(os.path.join(self.clip_dir, f"{mode}_clips.csv"), index=False)
        else:
            df = pd.read_csv(os.path.join(self.clip_dir, f"{mode}_clips.csv"))
            df['category'] = df['category'].apply(lambda x: str(x).split(','))
            for _, r in df.iterrows():
                self.meta[r['file_name']] = r['category']
        self.indices = list(self.meta.keys())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: Tuple[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns tensor of audio and its label."""
        y = ', '.join(self.meta[item])
        x = os.path.join(self.clip_dir, f'{item}.wav')
        if self.cfgs['data_type'] == 'audio':
            x = torchaudio.load(x, normalize=True)
        if self.cfgs['target_type'] != 'category':
            y = self.tokeniser[self.cfgs['target_type']]
        return x, y

    def _read_csv(self, csv_path: str) -> dict:
        r"""Load meta information from a csv file.
            Returns a dict: 'file_name' -> (class) 'category'.
        """
        meta = dict()
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            meta[str(r['file_name'])] = r['category'].split(',')
        return meta

    def _make_clips(self, seg_meta: dict) -> dict:
        r"""Prepare audio clips with fixed length and 
        returns clip-wise meta information: `clip_name` -> `clip_category`."""
        seg_names, seg_audios, seg_cats = list(), list(), list()
        for sname, scat in seg_meta.items():
            saudio, ori_sr = torchaudio.load(os.path.join(self.audio_dir, f"{sname}.wav"), normalize=True)
            seg_audios.append(saudio)
            seg_names.append(sname)
            seg_cats.append(scat)
        if ori_sr != FSD_FS.clip_cfgs['sample_rate']:
            _tmp = list()
            resample_fn = torchaudio.transforms.Resample(ori_sr, sr)
            for s in seg_audios:
                _tmp.append(resample_fn(s))
            seg_audios = _tmp
        # Clip segments into the fixed length
        clip_duration = int(FSD_FS.clip_cfgs['sample_rate'] * FSD_FS.clip_cfgs['duration'])
        clip_hop = int(clip_duration * FSD_FS.clip_cfgs['hop_length'])
        clip_names, clip_audios, clip_cats = list(), list(), list()
        for idx, seg in tqdm.tqdm(enumerate(seg_audios), total=len(seg_audios)):
            clips = self._clip_segment(wav=seg, win_length=clip_duration, hop_length=clip_hop)
            clip_audios.extend(clips)
            clip_names.extend([seg_names[idx] + f"_{i:03d}" for i, _ in enumerate(clips)])
            clip_cats.extend(seg_cats[idx] for _ in clips)
        assert len(clip_names) == len(clip_audios) == len(clip_cats)
        clip_data = {
            'clip_name': clip_names,
            'clip_audio': clip_audios,
            'clip_category': clip_cats
        }
        return clip_data

    def _clip_segment(self, wav: torch.Tensor, win_length: int, hop_length: int) -> List[torch.Tensor]:
        r"""Make variant-length a waveform into a fixed one by trancating and padding (replicating).
        wav is expected to be a channel_first tensor. """
        def _replicate(x: torch.Tensor, min_clip_duration: int) -> torch.Tensor:
            """ Pad a 1-D tensor to fix-length `min_clip_duration` by replicating the existing elements."""
            tile_size = (min_clip_duration // x.size(dim=-1)) + 1
            x = torch.tile(x, dims=(tile_size,))[:min_clip_duration]
            return x

        clips = list()
        if wav.size(dim=-1) < win_length:
            tmp = _replicate(wav.squeeze(), win_length) # transfer a mono 2-D waveform to 1-D tensor
            clips.append(tmp.unsqueeze(dim=0)) # recover the waveform into size = (n_channel=1, n_samples)
        else:
            for idx in range(0, len(wav), hop_length):
                tmp = wav[idx:idx + win_length]
                tmp = _replicate(tmp.squeeze(), win_length)  # to ensure the last seq have the same length
                clips.append(tmp.unsqueeze(dim=0))
        return clips

    def _batch_dump(self, audios: List[torch.Tensor], file_names: List[str]) -> None:
        r"""Dump a batch of audios separatively."""
        print(f"Store audio file(s) to {self.clip_dir}")
        for cname, caudio in zip(file_names, audios):
            torchaudio.save(
                filepath=os.path.join(self.clip_dir, f'{cname}.wav'),
                src=caudio,
                sample_rate=FSD_FS.clip_cfgs['sample_rate'],
                encoding='PCM_S'
            )

    def tokeniser(self) -> dict:
        r"""Returns a dict: 'class_category' -> dict(`class_id`, `class_mid`)"""
        csv_path = os.path.join(self.csv_dir, 'vocabulary.csv')
        df = pd.read_csv(csv_path, names=['id', 'category', 'mid'])
        tokeniser = dict()
        for idx, r in df.iterrows():
            tokeniser[r['category']] = {
                'id': r['id'],
                'mid': r['mid']
            }
        return tokeniser

    @property
    def labelset(self):
        labels = list()
        for lset in self.meta.values():
            labels.extend(lset)
        return list(set(labels))


class MLFewShotSampler():
    r"""A multi-label few-shot sampler.
    Args:
        dataset: data source, e.g. instance of torch.data.Dataset, data[item] = {filename: labels}.
        n_class: number of novel classes in an episode (i.e., 'n' ways).
        n_supports: number of support samples in each novel class (i.e., 'k' shot).
        n_queries: number of queries for each novel class.
        n_task: total number of tasks (or episodes) in one epoch.
    """
    def __init__(
            self,
            dataset: torch.nn.Module,
            labelset: list,
            n_class: int = 15,
            n_supports: int = 5,
            n_queries: int = 5,
            n_task: int = 100,
            **kargs
    ) -> None:
        self.dataset = dataset
        self.labelset = labelset
        self.n_cls = n_class
        self.n_supports = n_supports
        self.n_queries = n_queries
        self.n_task = n_task

    def __len__(self):
        return self.n_task

    def __iter__(self):
        r"""Returns a list of format: (`file_path`, `target_id`)."""
        for _ in range(self.n_task):
            batch_x, batch_y = list(), list()
            selected_classes = np.random.choice(self.labelset, size=self.n_cls, replace=False)
            # Create a data subset containing attached to novel classes ONLY
            subset = {'fpath': [], 'label': []}
            for fpath, lbl in self.dataset.meta.items():
                if np.any(np.isin(lbl, selected_classes)):
                    subset['fpath'].append(fpath)
                    subset['label'].append(lbl)
            # Sample support examples and assign with labels
            for n in selected_classes:
                _candidate = list()
                for fpath, lbl in zip(subset['fpath'], subset['label']):
                    if n in lbl:
                        _candidate.append(str(fpath))
                _samples = np.random.choice(_candidate, size=self.n_supports, replace=False)
                batch_x.extend(_samples.tolist())
            supports = batch_x  
            # Sample query examples and assign with labels
            for n in selected_classes:
                _candidate = list()
                for fpath, lbl in zip(subset['fpath'], subset['label']):
                    if n in lbl and (fpath not in supports):
                        _candidate.append(str(fpath))
                _samples = np.random.choice(_candidate, size=self.n_queries, replace=False)
                batch_x.extend(_samples.tolist())
            yield batch_x

if __name__ == '__main__':
    clip_dir = '/data/EECS-MachineListeningLab/datasets/FSD_FS/clips'
    audio_dir = '/data/EECS-MachineListeningLab/datasets/FSD_FS'
    csv_dir = '/data/EECS-MachineListeningLab/datasets/FSD_FS/meta'
    modes = ['base', 'val', 'eval']
    val = modes[1]

    fsdfs = FSD_FS(clip_dir=clip_dir, audio_dir=audio_dir, csv_dir=csv_dir, mode=val, data_type='path', target_type='category')
    tokeniser = fsdfs.tokeniser()
    detokeniser = dict()
    for cat, token in tokeniser.items():
        detokeniser[token['mid']] = cat
    selected_list = [detokeniser[mid] for mid in fsd50k_select_ids[val]]
    sampler = MLFewShotSampler(dataset=fsdfs, labelset=selected_list, n_class=15, n_supports=1, n_queries=5, n_task=100)
    dataloader = DataLoader(fsdfs, batch_sampler=sampler, num_workers=4, pin_memory=True)
    for x, y in dataloader:
        print(f"x={x}\n")
        print(f"target={y}\n")
