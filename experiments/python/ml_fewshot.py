import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import argparse
import hydra
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from typing import Optional, Callable
from omegaconf import OmegaConf
sys.path.insert(0, "../..")
from src.CLAPWrapper import CLAPWrapper
from src.data import prepare_data, SimpleFewShotSampler
from retrieve import compute_similarity
from utils import CustomDistance, confidence_interval, normc2d, tgt_tokenise_fn

torch.manual_seed(42)

class MultiLabelFewShot():
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
        tgt_tokeniser: Callable,
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
        self.tgt_tokeniser = tgt_tokeniser
        self.tgt_fmt = ', '

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
            _tmp_ys = list()
            for y in ys:
                _tmp_ys.append(y.split(', '))
            ys = _tmp_ys
            if self.fine_tune: 
                _map, _roc = self._adapt_on_batch(model=clap_model, wav_paths=xs, targets=ys, a=self.a, b=self.b, **self.fewshot_cfgs, **self.tr_cfgs)
            else:
                _map, _roc = self._test_on_batch(model=clap_model, wav_paths=xs, targets=ys, a=self.a, b=self.b, **self.fewshot_cfgs)
            map.append(_map)
            roc.append(_roc)
            if verbose:
                print(f"Running map: {_map}, roc: {_roc}")
        map = torch.tensor(map, dtype=torch.float).mean()
        roc = torch.tensor(roc, dtype=torch.float).mean()
        return map, roc

    def _adapt_on_batch(self, model: torch.nn.Module, wav_paths: list, targets: list, n_class: int, n_supports: int, n_queries: int, a: float, b: float, train_epochs: int, train_lr: float) -> Tensor:
        r"""Trainable version of few- & zero-shot classification."""
        # Generate a list of selected labels and corresponding captions
        labelset = list(set(targets))
        caps = [self.prompt + l for l in labelset]
        # CLAP forward
        with torch.no_grad():
            audio_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
            audio_embeddings = normc2d(audio_embeddings)  # normalise each column of audio embeddings
            text_embeddings = model.get_text_embeddings(caps)
        support_embeddings, query_embeddings = audio_embeddings[:n_supports * n_class], audio_embeddings[n_supports * n_class:]
        # Predict labels with affinity between support and query embeddings
        _support_targets, query_targets = targets[:n_supports * n_class], targets[n_supports * n_class:]
        support_onehots = self.tgt_tokeniser(target=_support_targets, labelset=labelset).to(device=query_embeddings.device)
        # Initialise adapter using support embeddings
        adapter = torch.nn.Linear(support_embeddings.size(dim=1), support_embeddings.size(dim=0), bias=False).to(device=audio_embeddings.device)  # dtype=model.clap.dtype, device=model.clap.device
        adapter.weight = torch.nn.Parameter(support_embeddings)
        # Another adapter to convert audio embedding
        # embed_dim = support_embeddings.size(dim=1)
        # adapter = torch.nn.Linear(embed_dim, embed_dim, bias=False).to(device=audio_embeddings.device)
        # adapter.weight = torch.nn.Parameter(torch.eye(embed_dim, device=audio_embeddings.device))

        optimiser = torch.optim.AdamW(adapter.parameters(), lr=0.0, eps=1e-4) # train_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, train_epochs)
        r"""Fine-tune adapter using training (support) data."""
        shuffle_idx = torch.randperm(support_embeddings.size(dim=0))
        train_emb, train_onehots = support_embeddings[shuffle_idx], support_onehots[shuffle_idx]
        adapter.train()
        for id in range(train_epochs):
            assert train_emb.size(dim=1) == support_embeddings.size(dim=1)
            _attention = adapter(train_emb)  # similarity(train_emb, support_embeddings)
            # Another adapter usage
            # _new_train_emb, _new_support_emb = adapter(train_emb), adapter(support_embeddings)
            # _attention = _new_train_emb @ _new_support_emb.T

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
        # attention = adapter(query_embeddings)
        new_query_emb, new_support_emb = adapter(query_embeddings), adapter(support_embeddings)
        attention = new_query_emb @ new_support_emb.T
        fewshot_logits = torch.exp(- b + b * attention) @ support_onehots

        zeroshot_logits = model.compute_similarity(query_embeddings, text_embeddings)
        preds = (a * fewshot_logits + zeroshot_logits).sigmoid()
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
        labelset = list()
        for t in targets:
            labelset.extend(t)
        labelset = list(set(labelset))
        caps = [self.prompt + l for l in labelset]
        # CLAP forward
        audio_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
        support_embeddings, query_embeddings = audio_embeddings[:n_supports * n_class], audio_embeddings[n_supports * n_class:]
        # Predict labels with affinity between support and query embeddings
        targets = self.tgt_tokeniser(target=targets, labelset=labelset).to(device=query_embeddings.device)
        support_targets, query_targets = targets[:n_supports * n_class], targets[n_supports * n_class:]
        fewshot_logits = self._affinity_predict(q_x=query_embeddings, s_x=support_embeddings, s_y=support_onehots, b=self.b)
        # Predict labels with similarity between audio and text embeddings
        text_embeddings = model.get_text_embeddings(caps)
        zeroshot_logits = model.compute_similarity(query_embeddings, text_embeddings)  # size = (n_wav, n_class)
        preds = (a * fewshot_logits + zeroshot_logits).sigmoid()
        # Compare predictions with query targets
        map = metrics.average_precision_score(query_targets, preds, average='macro')
        roc = metrics.roc_auc_score(labels, predictions, average='macro')
        return map, roc
    
    def _affinity_predict(self, q_x: Tensor, s_x: Tensor, s_y: Tensor, b: float) -> Tensor:
        r"""Predict query labels using affinity matrix between query and supports as:
            :math: `\text{logits} = \alpha As_y`
            :math: `A = \exp(-\beta d_{cos}(f(q_x), f(s_x)))`
        """
        attention = torch.exp(- b * self.distance_fn(q_x, s_x))
        return torch.mm(attention, s_y) 


@hydra.main(version_base=None, config_path='../cfgs', config_name='experiments_cfgs')
def main(cfgs: OmegaConf) -> None:
    print(f"The seetings for the experiment of {cfgs}.")
    # I/O
    db_name = cfgs['database']
    exp_type = cfgs['experiment']
    weights_pth = cfgs['model_weights_path']

    audio_dir = cfgs[db_name]['audio_dir']
    csv_path = cfgs[db_name]['csv_path']
    
    DataSet, Sampler, fs_label_splits = prepare_data(data_source=db_name)
    val_labelset = fs_label_splits[cfgs[db_name]['mode']]
    database = DataSet(
        audio_dir=audio_dir, 
        csv_path=csv_path, 
        data_type='path', 
        target_type='category',
        clip_dir=cfgs['fsd_fs']['clip_dir'],
        mode=cfgs['fsd_fs']['mode']
        )
    tgt_tokeniser = tgt_tokenise_fn('multihot')
    print(f"Experiment on {cfgs[db_name]['mode']} split.}")
    # train_labelset = [x for x in range(len(database.labelset)) if x not in val_labelset]
    sampler = Sampler(
        dataset=database, 
        labelset=val_labelset,
        n_class=cfgs['fewshot']['n_class'],
        n_supports=cfgs['fewshot']['n_supports'],
        n_queries=cfgs['fewshot']['n_queries'],
        n_task=100
        )
    dataloader = DataLoader(database, batch_sampler=sampler, num_workers=4, pin_memory=True)
    prompt = 'this is a sound of '
    r""" Now begin our experiment."""
    fewshot = MultiLabelFewShot(
        dataloader=dataloader, 
        weights_pth=weights_pth, 
        prompt=prompt, 
        n_class=cfgs['fewshot']['n_class'],
        n_supports=cfgs['fewshot']['n_supports'],
        n_queries=cfgs['fewshot']['n_queries'],
        a=cfgs['fewshot']['match']['a'],
        b=cfgs['fewshot']['match']['b'],
        distance='cosine', 
        cuda=True,
        tgt_tokeniser=tgt_tokeniser,
        fine_tune=cfgs['fewshot']['fine_tune'],
        train_epochs=cfgs['fewshot']['train_epochs'],
        train_lr=cfgs['fewshot']['learning_rate'],
        )
    map, roc = fewshot.forward()
    print(f"The results on {cfgs[db_name]['mode']}: map={map}, roc={roc}")


if __name__ == '__main__':
    main()