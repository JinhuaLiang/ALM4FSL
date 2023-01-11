import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import argparse
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
sys.path.insert(0, "../..")
from src.CLAPWrapper import CLAPWrapper
from src.data import prepare_data, SimpleFewShotSampler
from retrieve import compute_similarity
from utils import CustomDistance, confidence_interval, normc2d

torch.manual_seed(42)


def zero_shot(**cfgs) -> None:
    labelset = cfgs['database'].labelset
    wav_pths, tgts = list(), list()
    for x, y in cfgs['database']:
        wav_pths.append(x)
        tgts.append(y)
    caps = [cfgs['prompt'] + l for l in labelset]
    preds = compute_similarity(weights_pth, caps, wav_pths)
    # Compare predictions with targets
    preds = preds.argmax(dim=1).tolist()
    assert len(preds) == len(tgts)
    correct_cnt = 0
    for idx, p in tqdm(enumerate(preds)):
        if labelset[p] == tgts[idx]:
            correct_cnt += 1
    acc = correct_cnt / len(preds)
    print(f"Accuracy: {acc}")


class FewShot():
    def __init__(
        self, 
        dataloader: torch.nn.Module, 
        weights_pth: str, 
        prompt: str, 
        n_class: int, 
        n_supports: int, 
        n_queries: int, 
        a: float, 
        b: float, 
        distance: str,
        cuda: bool,
        fine_tune: bool = False,
        train_epochs: int = 20,
        train_lr: float = 1e-4,
    ) -> None:
        self.dataloader = dataloader
        self.weights_pth = weights_pth
        self.prompt = prompt
        self.fewshot_cfgs = {
            'n_class': n_class,
            'n_supports': n_supports,
            'n_queries': n_queries
        }
        self.a = a
        self.b = b
        self.distance_fn = CustomDistance(type=distance)
        self.cuda = cuda

        self.fine_tune = fine_tune
        if self.fine_tune:
            self.tr_cfgs = {
                'train_epochs': train_epochs,
                'train_lr': train_lr,
            }
        
    def forward(self, verbose: bool = False):
        clap_model = CLAPWrapper(self.weights_pth, use_cuda=self.cuda)
        acc = list()
        # for xs, ys in tqdm(self.dataloader):
        for xs, ys in self.dataloader:
            if self.fine_tune: 
                _running_acc = self._adapt_on_batch(model=clap_model, wav_paths=xs, targets=ys, a=self.a, b=self.b, **self.fewshot_cfgs, **self.tr_cfgs)
            else:
                _running_acc = self._test_on_batch(model=clap_model, wav_paths=xs, targets=ys, a=self.a, b=self.b, **self.fewshot_cfgs)
            acc.append(_running_acc)
            if verbose:
                print(f"Running accuracy: {_running_acc}")
        acc = torch.tensor(acc, dtype=torch.float).mean()
        return acc

    def _adapt_on_batch(self, model: torch.nn.Module, wav_paths: list, targets: list, n_class: int, n_supports: int, n_queries: int, a: float, b: float, train_epochs: int, train_lr: float) -> Tensor:
        r"""Trainable version of few- & zero-shot classification."""
        # Generate a list of selected labels and corresponding captions
        labelset = list(set(targets))
        caps = [prompt + l for l in labelset]
        # CLIP forward
        with torch.no_grad():
            audio_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
            audio_embeddings = normc2d(audio_embeddings)  # normalise each column of audio embeddings
            text_embeddings = model.get_text_embeddings(caps)
        support_embeddings, query_embeddings = audio_embeddings[:n_supports * n_class], audio_embeddings[n_supports * n_class:]
        # Predict labels with affinity between support and query embeddings
        _support_targets, query_targets = targets[:n_supports * n_class], targets[n_supports * n_class:]
        support_onehots = self._one_hot(target=_support_targets, labelset=labelset).to(device=query_embeddings.device)
        # Initialise adapter using support embeddings
        adapter = torch.nn.Linear(support_embeddings.size(dim=1), support_embeddings.size(dim=0), bias=False).to(device=audio_embeddings.device)  # dtype=model.clap.dtype, device=model.clap.device
        adapter.weight = torch.nn.Parameter(support_embeddings)
        # print(f"adapter.weight:{adapter.weight}")
        # print(f"support_embeddings:{support_embeddings}")
        optimiser = torch.optim.AdamW(adapter.parameters(), lr=train_lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, train_epochs)
        r"""Fine-tune adapter using training (support) data."""
        shuffle_idx = torch.randperm(support_embeddings.size(dim=0))
        train_emb, train_onehots = support_embeddings[shuffle_idx], support_onehots[shuffle_idx]
        adapter.train()
        for id in range(train_epochs):
            assert train_emb.size(dim=1) == support_embeddings.size(dim=1)
            _attention = adapter(train_emb)  # similarity(train_emb, support_embeddings)

            _fewshot_logits = torch.exp(- b + b * _attention) @ support_onehots
            _zeroshot_logits = model.compute_similarity(train_emb, text_embeddings)
            _overall_logits = (a * _fewshot_logits + _zeroshot_logits)

            loss = torch.nn.functional.cross_entropy(_overall_logits, train_onehots.argmax(dim=-1))
            # Update params
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
        r"""Evaluate adapter using eval (query) data."""
        adapter.eval()
        attention = adapter(query_embeddings)
        fewshot_logits = torch.exp(- b + b * attention) @ support_onehots

        zeroshot_logits = model.compute_similarity(query_embeddings, text_embeddings)
        preds = (a * fewshot_logits + zeroshot_logits)
        # Compare predictions with query targets
        preds = preds.argmax(dim=1).tolist()
        correct_cnt = 0
        for idx, p in enumerate(preds):
            if labelset[p] == query_targets[idx]:
                correct_cnt += 1
        _running_acc = correct_cnt / len(preds)
        return _running_acc

    def _test_on_batch(self, model: torch.nn.Module, wav_paths: list, targets: list, n_class: int, n_supports: int, n_queries: int, a: float, b: float) -> Tensor:
        r"""Predict query logits using support embeddings and targets."""
        # Generate a list of selected labels and corresponding captions
        labelset = list(set(targets))
        caps = [prompt + l for l in labelset]
        # CLIP forward
        audio_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
        support_embeddings, query_embeddings = audio_embeddings[:n_supports * n_class], audio_embeddings[n_supports * n_class:]
        # Predict labels with affinity between support and query embeddings
        support_targets, query_targets = targets[:n_supports * n_class], targets[n_supports * n_class:]
        support_onehots= self._one_hot(target=support_targets, labelset=labelset).to(device=query_embeddings.device)
        fewshot_logits = self._affinity_predict(q_x=query_embeddings, s_x=support_embeddings, s_y=support_onehots, b=self.b)
        # Predict labels with similarity between audio and text embeddings
        text_embeddings = model.get_text_embeddings(caps)
        zeroshot_logits = model.compute_similarity(query_embeddings, text_embeddings)  # size = (n_wav, n_class)
        preds = (a * fewshot_logits + zeroshot_logits).softmax(dim=-1)
        # Compare predictions with query targets
        preds = preds.argmax(dim=1).tolist()
        correct_cnt = 0
        for idx, p in enumerate(preds):
            if labelset[p] == query_targets[idx]:
                correct_cnt += 1
        _running_acc = correct_cnt / len(preds)
        return _running_acc
    
    def _affinity_predict(self, q_x: Tensor, s_x: Tensor, s_y: Tensor, b: float) -> Tensor:
        r"""Predict query labels using affinity matrix between query and supports as:
            :math: `\text{logits} = \alpha As_y`
            :math: `A = \exp(-\beta d_{cos}(f(q_x), f(s_x)))`
        """
        attention = torch.exp(- b * self.distance_fn(q_x, s_x))
        return torch.mm(attention, s_y) 

    def _one_hot(self, target: list, labelset: list) -> Tensor:
        r"""A simple way to generate one-hot label for few-shot algorithms.
        Returns: A tensor with shape = (len(target), len(labelset)).
        e.g., a = self._one_hot(target=['a', 'b'], labelset=['a', 'b', 'c'])  # [[1, 0, 0], [0, 1, 0]]
        """
        # Create dict: target -> class_id
        _tokeniser = dict()
        for idx, l in enumerate(labelset):
            _tokeniser[l] = idx
        one_hot = torch.zeros(len(target), len(labelset))
        for idx, tgt in enumerate(target):
            one_hot[idx][_tokeniser[tgt]] = 1
        return one_hot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="esc50")
    parser.add_argument("--n_class", type=int, default=15)
    parser.add_argument("--n_supports", type=int, default=1)
    parser.add_argument("--n_queries", type=int, default=30)
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--train_epochs", type=int, default=15)
    parser.add_argument("--train_lr", type=float, default=0.0001)
    args = parser.parse_args()
    # I/O
    weights_pth = '/data/EECS-MachineListeningLab/jinhua/AudioSSL/CLAP_weights_2022.pth'
    audio_dirs = {
        'esc50': '/data/EECS-MachineListeningLab/datasets/ESC-50/audio',
        'fsdkaggle18k': [
            '/data/EECS-MachineListeningLab/datasets/FSDKaggle2018/FSDKaggle2018.audio_train', 
            '/data/EECS-MachineListeningLab/datasets/FSDKaggle2018/FSDKaggle2018.audio_test'
            ]
    }
    csv_paths = {
        'esc50': '/data/EECS-MachineListeningLab/datasets/ESC-50/meta/esc50.csv',
        'fsdkaggle18k': [
            '/data/EECS-MachineListeningLab/datasets/FSDKaggle2018/FSDKaggle2018.meta/train_post_competition.csv', 
            '/data/EECS-MachineListeningLab/datasets/FSDKaggle2018/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv'
            ]
    }
    DataBase, fs_label_splits = prepare_data(data_source=args.dataset)
    database = DataBase(
        audio_dir=audio_dirs[args.dataset], 
        csv_path=csv_paths[args.dataset], 
        data_type='path', 
        target_type='category'
        )
    # Exp setting
    fewshot_cfgs = {
        'n_class': args.n_class, 
        'n_supports': args.n_supports, 
        'n_queries': args.n_queries
    }
    finetune_cfgs = {
        'fine_tune': args.fine_tune,
        'train_epochs': args.train_epochs,
        'train_lr': args.train_lr
    }
    print(f"The seetings for the experiment of {args.dataset}: {fewshot_cfgs}, {finetune_cfgs}")
    history = list()
    for idx, val_labelset in enumerate(fs_label_splits):
        print(f"Cross-validation: {idx}/{len(fs_label_splits)}")
        train_labelset = [x for x in range(len(database.labelset)) if x not in val_labelset]
        sampler = SimpleFewShotSampler(dataset=database, labelset=val_labelset, n_task=100, **fewshot_cfgs)
        dataloader = DataLoader(database, batch_sampler=sampler, num_workers=4, pin_memory=True)
        prompt = 'this is a sound of '
        r"""Now begin our experiment"""
        fewshot = FewShot(dataloader=dataloader, weights_pth=weights_pth, prompt=prompt, a=1.0, b=5.5, distance='cosine', cuda=True, **fewshot_cfgs, **finetune_cfgs)
        _acc = fewshot.forward()
        history.append(_acc)
        print(f"Final accuracy={_acc}")
    mean, interval = confidence_interval(x=np.asarray(history), confidence=0.95)
    print(f"The {len(fs_label_splits)}-fold cross-validation: mean={mean}, var={interval}.")
