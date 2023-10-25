import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Callable
from torchaudio.transforms import Spectrogram
from functools import reduce, partial


class MLC(nn.Module):
    def __init__(
        self,
        n_fft: int,
        sr: int,
        gammas: List[float],
        hop_size: int,
        Hipass_f: float = 50,
        Lowpass_t=0.24,
        **kwargs,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size

        self.stft = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_size,
            power=2,
            normalized=True,
            onesided=False,
            **kwargs,
        )
        hpi = int(Hipass_f * n_fft / sr) + 1
        lpi = int(Lowpass_t * sr / 1000) + 1
        layers = [lambda ceps, spec: (ceps, spec ** gammas[0])]

        def gamm_trsfm(x, g, i):
            x = torch.fft.fft(x, norm="ortho").real
            x[..., :i] = x[..., -i:] = 0
            return F.relu(x) ** g

        self.num_spec = 1
        self.num_ceps = 0

        for d, gamma in enumerate(gammas[1:]):
            if d % 2:
                layers.append(lambda ceps, _: (ceps, gamm_trsfm(ceps, gamma, hpi)))
                self.num_spec += 1
            else:
                layers.append(lambda _, spec: (gamm_trsfm(spec, gamma, lpi), spec))
                self.num_ceps += 1

        self.compute = partial(reduce, lambda x, f: f(*x), layers)

    def forward(self, x):
        return self.compute((None, self.stft(x).transpose(-1, -2)))


class Sparse_Pitch_Profile(nn.Module):
    def __init__(self, in_channels, sr, harms_range=24, division=1, norm=False):
        """

        Parameters
        ----------
        in_channels: int
            window size
        sr: int
            sample rate
        harms_range: int
            The extended area above (or below) the piano pitch range (in semitones)
            25 : though somewhat larger, to ensure the coverage is large enough (if division=1, 24 is sufficient)
        division: int
            The division number for filterbank frequency resolution. The frequency resolution is 1 / division (semitone)
        norm: bool
            If set to True, normalize each filterbank so the weight of each filterbank sum to 1.
        """
        super().__init__()
        step = 1 / division
        # midi_num shape = (88 + harms_range) * division + 2
        # this implementation make sure if we group midi_num with a size of division
        # each group will center at the piano pitch number and the extra pitch range
        # E.g., division = 2, midi_num = [20.25, 20.75, 21.25, ....]
        #       dividion = 3, midi_num = [20.33, 20.67, 21, 21.33, ...]
        midi_num = np.arange(
            20.5 - step / 2 - harms_range, 108.5 + step + harms_range, step
        )
        self.midi_num = midi_num

        fd = 440 * np.power(2, (midi_num - 69) / 12)
        self.fd = fd

        self.effected_dim = in_channels // 2 + 1
        # // 2 : the spectrum/ cepstrum are symmetric

        x = np.arange(self.effected_dim)
        freq_f = x * sr / in_channels
        freq_t = sr / x[1:]
        # avoid explosion; x[0] is always 0 for cepstrum

        inter_value = np.array([0, 1, 0])
        idxs = np.digitize(freq_f, fd)

        cols, rows, values = [], [], []
        for i in range(harms_range * division, (88 + 2 * harms_range) * division):
            idx = np.where((idxs == i + 1) | (idxs == i + 2))[0]
            c = idx
            r = np.broadcast_to(i - harms_range * division, idx.shape)
            x = np.interp(freq_f[idx], fd[i : i + 3], inter_value).astype(np.float32)
            if norm and len(idx):
                # x /= (fd[i + 2] - fd[i]) / sr * in_channels
                x /= x.sum()  # energy normalization

            if len(idx) == 0 and len(values) and len(values[-1]):
                # low resolution in the lower frequency (for spec)/ highter frequency (for ceps),
                # some filterbanks will not get any bin index, so we copy the indexes from the previous iteration
                c = cols[-1].copy()
                r = rows[-1].copy()
                r[:] = i - harms_range * division
                x = values[-1].copy()

            cols.append(c)
            rows.append(r)
            values.append(x)

        cols, rows, values = (
            np.concatenate(cols),
            np.concatenate(rows),
            np.concatenate(values),
        )
        self.filters_f_idx = (rows, cols)
        self.filters_f_values = nn.Parameter(torch.tensor(values), requires_grad=False)

        idxs = np.digitize(freq_t, fd)
        cols, rows, values = [], [], []
        for i in range((88 + harms_range) * division - 1, -1, -1):
            idx = np.where((idxs == i + 1) | (idxs == i + 2))[0]
            c = idx + 1
            r = np.broadcast_to(i, idx.shape)
            x = np.interp(freq_t[idx], fd[i : i + 3], inter_value).astype(np.float32)
            if norm and len(idx):
                # x /= (1 / fd[i] - 1 / fd[i + 2]) * sr
                x /= x.sum()

            if len(idx) == 0 and len(values) and len(values[-1]):
                c = cols[-1].copy()
                r = rows[-1].copy()
                r[:] = i
                x = values[-1].copy()

            cols.append(c)
            rows.append(r)
            values.append(x)

        cols, rows, values = (
            np.concatenate(cols),
            np.concatenate(rows),
            np.concatenate(values),
        )
        self.filters_t_idx = (rows, cols)
        self.filters_t_values = nn.Parameter(torch.tensor(values), requires_grad=False)
        self.filter_size = torch.Size(
            ((88 + harms_range) * division, self.effected_dim)
        )

    def forward(self, ceps, spec):
        ceps, spec = ceps[..., : self.effected_dim], spec[..., : self.effected_dim]
        batch_dim, steps, _ = ceps.size()
        filter_f = torch.sparse_coo_tensor(
            self.filters_f_idx, self.filters_f_values, self.filter_size
        )
        filter_t = torch.sparse_coo_tensor(
            self.filters_t_idx, self.filters_t_values, self.filter_size
        )
        ppt = filter_t @ ceps.transpose(0, 2).contiguous().view(self.effected_dim, -1)
        ppf = filter_f @ spec.transpose(0, 2).contiguous().view(self.effected_dim, -1)
        return ppt.view(-1, steps, batch_dim).transpose(0, 2), ppf.view(
            -1, steps, batch_dim
        ).transpose(0, 2)
