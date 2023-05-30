import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.datasets.vctk import VCTK_092
from torchaudio.functional import resample


def vctk_collate(batch,
                 target_rate=22050,
                 length=32768):
    def pad_tensor(t):
        current_length = t.shape[1]
        padding_length = length - current_length
        return F.pad(t, (0, padding_length))

    original_sr = batch[0][1]
    waveforms = [pad_tensor(resample(waveform=data[0],
                                     orig_freq=original_sr,
                                     new_freq=target_rate)) for data in batch]
    return torch.stack(waveforms)


if __name__ == '__main__':

    dataset = VCTK_092(root="/home/emilian/PycharmProjects/multi-speaker-diff-sep/data/vctk", download=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=16,
                            collate_fn=vctk_collate,
                            pin_memory=True,
                            num_workers=0)
    output = next(iter(dataloader))