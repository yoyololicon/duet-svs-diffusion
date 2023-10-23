import os
from typing import Optional

import numpy as np
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchaudio
from pathlib import Path
from torchaudio.transforms import Resample

class WAVDataset(Dataset):
    """
    Wave-file-only Dataset.
    """

    def __init__(
        self,
        data_dir: str,
        segment: int,
        overlap: int = 0,
        keepdim: bool = False,
        mono: bool = False,
        resample: Optional[int] = None
    ):
        assert segment > overlap and overlap >= 0
        self.segment = segment
        self.data_path = os.path.expanduser(data_dir)
        self.hop_size = segment - overlap
        self.keepdim = keepdim
        self.mono = mono
        self.resample = resample
        self.waves = []
        self.sr = None
        self.files = []

        file_chunks = []

        print("Gathering training files ...")
        for filename in tqdm(sorted(Path(self.data_path).glob("**/*.wav"))):
            meta = torchaudio.info(filename)
            self.files.append(filename)
            file_chunks.append(max(0, meta.num_frames - segment) // self.hop_size + 1)

            if not self.sr:
                self.sr = meta.sample_rate
                if self.resample:
                    self.resample_transform = Resample(orig_freq=self.sr, new_freq=self.resample)
                else:
                    self.resample_transform = None
            else:
                assert meta.sample_rate == self.sr

        self.size = sum(file_chunks)
        self.boundaries = np.cumsum(np.array([0] + file_chunks))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        f = self.files[bin_pos]
        offset = (index - self.boundaries[bin_pos]) * self.hop_size
        x = torchaudio.load(f, frame_offset=offset, num_frames=self.segment)[0]
        # this should not happen, but I just want to make sure
        if x.shape[1] < self.segment:
            tmp = x.new_zeros([x.shape[0], self.segment])
            rand_offset = random.randint(0, self.segment - x.shape[1])
            tmp[:, rand_offset : rand_offset + x.shape[1]] = x
            x = tmp

        if self.resample is not None:
            x = self.resample_transform(x)

        x = (lambda x: x.squeeze(0) if not self.keepdim else x)(
            (lambda x: x.mean(0, keepdim=True) if self.mono else x)(x)
        ).permute(1, 0).numpy()
        # print(f"{x.shape = }")
        return x




if __name__ == '__main__':
    torchaudio.set_audio_backend('sox_io')

