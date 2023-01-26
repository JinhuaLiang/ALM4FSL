import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import argparse
import hydra
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Callable
from omegaconf import OmegaConf
sys.path.insert(0, "../..")
from src.CLAPWrapper import CLAPWrapper
from src.data import prepare_data, SimpleFewShotSampler
from retrieve import compute_similarity
from utils import CustomDistance, confidence_interval, normc2d, tgt_tokenise_fn

torch.manual_seed(42)


class ESC50FullsizeEvalution():
    def __init__(
        self, 
        train_dataloader: torch.nn.Module,
        eval_dataloader: torch.nn.Module,
        weights_pth: str, 
        prompt: str, 
        n_class: int, 
        n_supports: int, 
        n_queries: int, 
        a: float, 
        b: float, 
        distance: str,
        cuda: bool,
        tgt_tokeniser: Callable,
        fine_tune: bool = False,
        adapter_type: str = 'match', # xattention
        xatt_disturb: bool = True,
        train_epochs: int = 20,
        train_lr: float = 1e-4,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
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
        self.tgt_tokeniser = tgt_tokeniser

        self.fine_tune = fine_tune
        if self.fine_tune:
            self.tr_cfgs = {
                'train_epochs': train_epochs,
                'train_lr': train_lr,
            }
            self.adapter_type = adapter_type
            self.xatt_disturb = xatt_disturb
        
    def forward(self, verbose: bool = False):
        clap_model = CLAPWrapper(self.weights_pth, use_cuda=self.cuda)
        classwise_res, history = dict(), dict()
        for lbl in self.eval_dataloader.dataset.labelset:  # record results as per class
            history[lbl] = list()
        for xs, ys in self.train_dataloader:
            if self.fine_tune:
                _cls_acc = self._adapt_on_batch(model=clap_model, eval_dataloader=self.eval_dataloader, wav_paths=xs, targets=ys, a=self.a, b=self.b, **self.fewshot_cfgs, **self.tr_cfgs)
            else:
                _cls_acc = self._test_on_batch(model=clap_model, eval_dataloader=self.eval_dataloader, wav_paths=xs, targets=ys, a=self.a, b=self.b, **self.fewshot_cfgs)
            if verbose:
                print(f"Running class-wise accuracy: {_cls_acc}")
            for cat, v in _cls_acc.items():
                history[cat].append(v)
        for cat, cnt in history.items():
            classwise_res[cat] = (torch.tensor(cnt, dtype=torch.float).mean() / 8).detach().numpy()  # each label have 8 examples in one fold
        return classwise_res, history

    def _adapt_on_batch(self, model: torch.nn.Module, eval_dataloader: torch.nn.Module, wav_paths: list, targets: list, n_class: int, n_supports: int, n_queries: int, a: float, b: float, train_epochs: int, train_lr: float) -> Tensor:
        r"""Trainable version of few- & zero-shot classification."""
        # Generate a list of selected labels and corresponding captions
        labelset = list(set(targets))
        caps = [self.prompt + l for l in labelset]
        # CLAP forward
        with torch.no_grad():
            support_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
            support_embeddings = normc2d(support_embeddings)  # normalise each column of audio embeddings
            text_embeddings = model.get_text_embeddings(caps)
        support_onehots = self.tgt_tokeniser(target=targets, labelset=labelset).to(device=support_embeddings.device)
        # Initialise adapter using support embeddings
        if self.adapter_type == 'match':
            adapter = torch.nn.Linear(support_embeddings.size(dim=1), support_embeddings.size(dim=0), bias=False).to(device=support_embeddings.device)
            adapter.weight = torch.nn.Parameter(support_embeddings)
        elif self.adapter_type == 'xattention':
            embed_dim = support_embeddings.size(dim=1)
            adapter = torch.nn.Linear(embed_dim, embed_dim, bias=False).to(device=support_embeddings.device)
            if self.xatt_disturb:
                init_w = torch.eye(embed_dim) + 1e-4 * (torch.rand((embed_dim, embed_dim)) - 0.5)
            else:
                init_w = torch.eye(embed_dim)
            adapter.weight = torch.nn.Parameter(init_w.to(support_embeddings.device))

        optimiser = torch.optim.AdamW(adapter.parameters(), lr=train_lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, train_epochs)
        r"""Fine-tune adapter using training (support) data."""
        shuffle_idx = torch.randperm(support_embeddings.size(dim=0))
        train_emb, train_onehots = support_embeddings[shuffle_idx], support_onehots[shuffle_idx]
        adapter.train()
        for id in range(train_epochs):
            assert train_emb.size(dim=1) == support_embeddings.size(dim=1)
            if self.adapter_type == 'match':
                _attention = adapter(train_emb)
            elif self.adapter_type == 'xattention':
                _new_train_emb, _new_support_emb = adapter(train_emb), adapter(support_embeddings)
                _attention = _new_train_emb @ _new_support_emb.T

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
        history = dict(zip(labelset, [0 for _ in labelset]))  # dict: label -> count
        for q_pths, q_tgts in eval_dataloader:
            query_embeddings = model.get_audio_embeddings(q_pths, resample=False)
            query_embeddings = normc2d(query_embeddings)
            if self.adapter_type == 'match':
                attention = adapter(query_embeddings)
            elif self.adapter_type == 'xattention':
                new_query_emb, new_support_emb = adapter(query_embeddings), adapter(support_embeddings)
                attention = new_query_emb @ new_support_emb.T
        fewshot_logits = torch.exp(- b + b * attention) @ support_onehots

        zeroshot_logits = model.compute_similarity(query_embeddings, text_embeddings)
        preds = (a * fewshot_logits + zeroshot_logits)  # todo softmax
        preds = preds.argmax(dim=1).tolist()
        # Compare predictions with query targets
        for idx, p in enumerate(preds):
            if labelset[p] == q_tgts[idx]:
                history[labelset[p]] += 1
        return history

    @torch.no_grad()
    def _test_on_batch(self, model: torch.nn.Module, eval_dataloader: torch.nn.Module, wav_paths: list, targets: list, n_class: int, n_supports: int, n_queries: int, a: float, b: float) -> Tensor:
        r"""Predict query logits using support embeddings and targets."""
        # Generate a list of selected labels and corresponding captions
        labelset = list(set(targets))
        caps = [self.prompt + l for l in labelset]
        # CLIP forward
        text_embeddings = model.get_text_embeddings(caps)
        support_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
        support_onehots= self.tgt_tokeniser(target=targets, labelset=labelset).to(device=support_embeddings.device)
        history = dict(zip(labelset, [0 for _ in labelset]))  # dict: label -> count
        for q_pths, q_tgts in eval_dataloader:
            query_embeddings = model.get_audio_embeddings(q_pths, resample=False)
            fewshot_logits = self._affinity_predict(q_x=query_embeddings, s_x=support_embeddings, s_y=support_onehots, b=self.b)
            # Predict labels with similarity between audio and text embeddings
            zeroshot_logits = model.compute_similarity(query_embeddings, text_embeddings)
            preds = (a * fewshot_logits + zeroshot_logits).softmax(dim=-1)
            preds = preds.argmax(dim=1).tolist()
            for idx, p in enumerate(preds):
                if labelset[p] == q_tgts[idx]:
                    history[labelset[p]] += 1
        return history

    
    def _affinity_predict(self, q_x: Tensor, s_x: Tensor, s_y: Tensor, b: float) -> Tensor:
        r"""Predict query labels using affinity matrix between query and supports as:
            :math: `\text{logits} = \alpha As_y`
            :math: `A = \exp(-\beta d_{cos}(f(q_x), f(s_x)))`
        """
        attention = torch.exp(- b * self.distance_fn(q_x, s_x))
        return torch.mm(attention, s_y) 


@hydra.main(version_base=None, config_path='../cfgs', config_name='esc50_fullsize')
def main(cfgs: OmegaConf) -> None:
    print(f"The seetings for the experiment of {cfgs}.")
    # I/O
    db_name = cfgs['database']
    weights_pth = cfgs['model_weights_path']

    audio_dir = cfgs[db_name]['audio_dir']
    csv_path = cfgs[db_name]['csv_path']
    
    history = dict()
    for fold in range(1, 6):
        print(f"Cross-validation: {fold}/5")
        DataSet, Sampler, fs_label_splits = prepare_data(data_source=db_name)
        tgt_tokeniser = tgt_tokenise_fn('onehot')
        train_database = DataSet(
            audio_dir=audio_dir, 
            csv_path=csv_path, 
            fold=[f for f in range(1, 6) if f != fold],
            data_type='path', 
            target_type='category',
            )
        eval_database = DataSet(
            audio_dir=audio_dir, 
            csv_path=csv_path, 
            fold=[fold],
            data_type='path', 
            target_type='category',
            )
        val_labelset = list(range(50))
        train_sampler = Sampler(
            dataset=train_database, 
            labelset=val_labelset,
            n_class=cfgs['fewshot']['n_class'],
            n_supports=cfgs['fewshot']['n_supports'],
            n_queries=0,
            n_task=cfgs['fewshot']['n_task']
            )
        eval_sampler = Sampler(
            dataset=eval_database, 
            labelset=val_labelset,
            n_class=cfgs['fewshot']['n_class'],
            n_supports=0,
            n_queries=cfgs['fewshot']['n_queries'],
            n_task=cfgs['fewshot']['n_task']
            )
        train_dataloader = DataLoader(train_database, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
        eval_dataloader = DataLoader(eval_database, batch_sampler=eval_sampler, num_workers=4, pin_memory=True)
        prompt = 'this is a sound of '
        r""" Now begin our experiment."""
        fewshot = ESC50FullsizeEvalution(
            train_dataloader=train_dataloader, 
            eval_dataloader=eval_dataloader,
            weights_pth=weights_pth, 
            prompt=prompt, 
            n_class=cfgs['fewshot']['n_class'],
            n_supports=cfgs['fewshot']['n_supports'],
            n_queries=cfgs['fewshot']['n_queries'],
            a=cfgs['fewshot']['a'],
            b=cfgs['fewshot']['b'],
            distance='cosine', 
            cuda=True,
            tgt_tokeniser=tgt_tokeniser,
            fine_tune=cfgs['fewshot']['fine_tune'],
            adapter_type=cfgs['fewshot']['adapter'],
            xatt_disturb=cfgs['fewshot']['xattention']['disturb'],
            train_epochs=cfgs['fewshot']['train_epochs'],
            train_lr=cfgs['fewshot']['learning_rate'],
            )
        _res, _ = fewshot.forward()
        for cat, acc in _res.items():
            print(f"Class {cat}'s accuracy={acc}")
            try:
                history[cat].append(acc)
            except:
                history[cat] = [acc]
    print(r"======")
    print(r"Summary on 5-fold validation:")
    overall_acc = list()
    for cat, acc in history.items():
        mean, interval = confidence_interval(x=np.stack(acc), confidence=0.95)
        overall_acc.append(mean)
        print(f"Acc. of class {cat}={mean} +- {interval}.")
    overall_acc = np.stack(overall_acc).mean()
    print(f"Overall accuracy = {overall_acc}")
    print(r"======")

if __name__ == '__main__':
    main()