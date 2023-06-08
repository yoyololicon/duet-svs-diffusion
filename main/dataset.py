import random
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.datasets.vctk import VCTK_092
from torchaudio.functional import resample


def vctk_collate(batch,
                 target_rate=22050,
                 length=32768,
                 mix_k = None):
    def pad_tensor(t):
        current_length = t.shape[1]
        padding_length = length - current_length
        return F.pad(t, (0, padding_length))

    original_sr = batch[0][1]
    waveforms = [pad_tensor(resample(waveform=data[0],
                                     orig_freq=original_sr,
                                     new_freq=target_rate)) for data in batch]
    if not mix_k:
        return torch.stack(waveforms)
    if len(waveforms) % mix_k != 0:
        raise ValueError("Batch size must be divisble by mix_k")
    random.shuffle(waveforms)
    partitioned_waveforms = [waveforms[i:i + mix_k] for i in range(0, len(waveforms), mix_k)]
    summed_list = [torch.sum(torch.stack(subset), dim=0) for subset in partitioned_waveforms]
    return torch.stack(summed_list)


if __name__ == '__main__':

    dataset = VCTK_092(root="/home/emilian/PycharmProjects/multi-speaker-diff-sep/data/vctk", download=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=16,
                            collate_fn=partial(vctk_collate, mix_k=2),
                            pin_memory=True,
                            num_workers=0)
    output = next(iter(dataloader))