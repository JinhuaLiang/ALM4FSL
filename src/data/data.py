from typing import Tuple
from torch.nn import Module
from .naive_dataset import SimpleFewShotSampler
from .esc50 import ESC50, fs_label_splits
from .fsdkaggle18k import FSDKaggle18K, fsdkaggle18k_val_labelsets


def prepare_data(data_source: str) -> Tuple[Module, list]:
    r"""Returns a dataloader and a set of novel class labels."""
    if data_source == "esc50":
        return ESC50, fs_label_splits
    elif data_source == "fsdkaggle18k":
        return FSDKaggle18K, fsdkaggle18k_val_labelsets
    else:
        raise ValueError(f"Cannot find a datasource name {data_source}.")
