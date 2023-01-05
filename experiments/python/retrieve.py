import os
import sys
import torch

sys.path.insert(0, "../..")
from src.CLAPWrapper import CLAPWrapper


def compute_similarity(weights_path: str, class_labels: list, wav_paths: list, *, use_cuda: bool = False) -> torch.Tensor:
    r"""Compute similarity score using CLAP."""
    clap_model = CLAPWrapper(weights_path, use_cuda=use_cuda)
    text_embeddings = clap_model.get_text_embeddings(class_labels)
    audio_embeddings = clap_model.get_audio_embeddings(wav_paths, resample=False)
    # size of similarity = (n_wav, n_class)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    return similarity.softmax(dim=-1)

if __name__ == "__main__":
    audio_dir = '/data/EECS-MachineListeningLab/datasets/ESC-50/audio'
    wav1 = os.path.join(audio_dir, '1-100032-A-0.wav')
    wav2 = os.path.join(audio_dir, '1-100038-A-14.wav')
    wav3 = os.path.join(audio_dir, '1-100210-A-36.wav')  # vacuum_cleaner
    weights_pth = '/data/EECS-MachineListeningLab/jinhua/AudioSSL/CLAP_weights_2022.pth'
    class_labels = ['dog', 'chirping_birds', 'vacuum_cleaner']
    wav_pths = [wav1, wav2, wav3]
    res = compute_similarity(weights_pth, class_labels, wav_pths)

    print(f"The similarity score is: {res}")
