import argparse
import hydra
import os
import torch
import pathlib
from tqdm import tqdm
from typing import Callable, Optional
from itertools import combinations, accumulate
from functools import reduce
import numpy as np
from math import sqrt
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import soundfile as sf
from torchnmf.nmf import NMF
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram, InverseSpectrogram

from mlc import Sparse_Pitch_Profile, MLC
from medley_vox import MedleyVox
from eval import COMPUTE_METRICS


def get_harmonics(n_fft, freqs, window_fn=torch.hann_window):
    window = window_fn(n_fft)
    tmp = []
    for f in freqs:
        h = torch.arange(int(0.5 * sr / f)) + 1
        # h = h[:40]
        ch = (
            f
            * 27
            / 4
            * (
                torch.exp(-1j * torch.pi * h)
                + 2 * (1 + 2 * torch.exp(-1j * torch.pi * h)) / (1j * torch.pi * h)
                - 6 * (1 - torch.exp(-1j * torch.pi * h)) / (1j * torch.pi * h) ** 2
            )
        )
        ch /= torch.abs(ch).max()
        t = torch.arange(n_fft) / sr
        eu = ch @ torch.exp(2j * torch.pi * h[:, None] * f * t)
        eu /= torch.linalg.norm(eu)
        tmp.append(torch.abs(torch.fft.fft(eu * window.numpy())[: n_fft // 2 + 1]))

    noise = torch.ones(n_fft // 2 + 1)
    tmp.append(noise)

    W_f0 = torch.stack(tmp).T
    W_f0 /= W_f0.max(0).values
    return W_f0


def get_streams(
    cfp: torch.Tensor,
    freqs: np.ndarray,
    n: int = 2,
    thresh_ratio: float = 0.1,
):
    top_values, top_indices = torch.topk(cfp, n * 3, dim=1)
    thresh = top_values.max() * thresh_ratio
    top_freqs = freqs[top_indices]

    # remove zero values
    top_zipped = [
        tuple(x[1] for x in filter(lambda x: x[0] > thresh, zip(*pair)))
        for pair in zip(top_values.tolist(), top_freqs.tolist())
    ]

    # init states

    def cases(states: tuple, possible_states: tuple):
        possible_states = tuple(sorted(possible_states))
        is_none = lambda x: x is None
        none_mapper = map(is_none, states)
        none_states = list(filter(is_none, states))
        valid_states = list(filter(lambda x: not is_none(x), states))

        if len(valid_states) == 0:
            return possible_states[:n] + (None,) * max(0, n - len(possible_states))
        elif len(possible_states) == 0:
            return (None,) * n

        def get_dist(curr, incoming):
            diff = abs(curr - incoming)
            return diff**2

        # first, possible_states are more than valid_states
        if len(possible_states) > len(valid_states):
            permutes = list(
                combinations(range(len(possible_states)), len(valid_states))
            )
            permuted_states = [
                [possible_states[i] for i in permute] for permute in permutes
            ]
            dists = map(
                lambda permute: reduce(
                    lambda acc, pair: acc + get_dist(*pair),
                    zip(valid_states, permute),
                    0,
                ),
                permuted_states,
            )
            min_permute_index = min(zip(dists, permutes), key=lambda x: x[0])[1]
            new_valid_states = tuple(possible_states[i] for i in min_permute_index)
            new_none_states = tuple(
                possible_states[i]
                for i in range(len(possible_states))
                if i not in min_permute_index
            )
        else:
            permutes = list(
                combinations(range(len(valid_states)), len(possible_states))
            )
            permuted_valid_states = [
                [valid_states[i] for i in permute] for permute in permutes
            ]
            dists = map(
                lambda permute: reduce(
                    lambda acc, pair: acc + get_dist(*pair),
                    zip(permute, possible_states),
                    0,
                ),
                permuted_valid_states,
            )
            min_permute_index = min(zip(dists, permutes), key=lambda x: x[0])[1]
            new_valid_states = ()
            for i in range(len(valid_states)):
                if i in min_permute_index:
                    new_valid_states += (possible_states[min_permute_index.index(i)],)
                else:
                    new_valid_states += (None,)

            new_none_states = ()
        new_none_states = (
            new_none_states[: len(none_states)]
            if len(new_none_states) > len(none_states)
            else new_none_states + (None,) * (len(none_states) - len(new_none_states))
        )

        # merge states
        new_states, *_ = reduce(
            lambda acc, is_none: (acc[0] + acc[1][:1], acc[1][1:], acc[2])
            if is_none
            else (acc[0] + acc[2][:1], acc[1], acc[2][1:]),
            none_mapper,
            ((), new_none_states, new_valid_states),
        )
        return new_states

    state_changes = list(accumulate(top_zipped, func=cases, initial=(None,) * n))[1:]

    return list(zip(*state_changes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("medleyvox", type=str, help="Path to MedleyVox dataset")
    parser.add_argument("--n_fft", default=8192, type=int, help="N FFT")
    parser.add_argument("--hop_length", default=256, type=int, help="Hop size")
    parser.add_argument("--win_length", default=2048, type=int, help="Window size")
    parser.add_argument(
        "--window",
        choices=["hann", "blackman", "hamming"],
        help="Window type",
        default="hann",
    )
    parser.add_argument("--out", type=str, help="Output directory")
    parser.add_argument("--full-duet", action="store_true", help="Drop duet songs")
    parser.add_argument("--divisions", type=int, default=8, help="Divisions")
    parser.add_argument("--min-f0", type=float, default=50, help="Min F0")
    parser.add_argument("--max-f0", type=float, default=1000, help="Max F0")
    parser.add_argument(
        "--gammas", type=float, nargs="+", help="Gammas", default=[0.2, 0.6, 0.8]
    )
    parser.add_argument(
        "--thresh", type=float, default=0.0, help="Salience threshold ratio"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Beta")
    parser.add_argument("--kernel-size", type=int, default=3, help="Kernel size")

    args = parser.parse_args()

    window_fn = {
        "hann": torch.hann_window,
        "blackman": torch.blackman_window,
        "hamming": torch.hamming_window,
    }[args.window]

    sr = 24000

    dataset = MedleyVox(
        args.medleyvox,
        sample_rate=sr,
        drop_duet=not args.full_duet,
    )

    mlc = MLC(
        args.n_fft,
        sr,
        args.gammas,
        args.hop_length,
        win_length=args.win_length,
        window_fn=window_fn,
    )

    pitch_profiler = Sparse_Pitch_Profile(
        args.n_fft, sr, 0, division=args.divisions, norm=True
    )
    freqs = pitch_profiler.fd[1:-1]

    idx_low = (freqs >= args.min_f0).nonzero()[0][0]
    idx_high = (freqs <= args.max_f0).nonzero()[0][-1]
    selection_slice = slice(idx_low, idx_high + 1)
    freqs = freqs[selection_slice]

    W_f0 = get_harmonics(args.win_length, freqs, window_fn=window_fn)

    spec = Spectrogram(
        n_fft=args.win_length,
        hop_length=args.hop_length,
        power=None,
        window_fn=window_fn,
    )
    inv_spec = InverseSpectrogram(
        n_fft=args.win_length,
        hop_length=args.hop_length,
        window_fn=window_fn,
    )

    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    accumulate_metrics_mean = {}

    with tqdm(dataset) as pbar:
        for mix_num, (x, y, ids) in enumerate(pbar):
            ppt, ppf = pitch_profiler(*mlc(x.unsqueeze(0)))

            # smoothing
            kernel_size = args.kernel_size
            ppt = F.avg_pool1d(ppt, kernel_size, stride=1, padding=kernel_size // 2)
            ppf = F.avg_pool1d(ppf, kernel_size, stride=1, padding=kernel_size // 2)

            cfp = (ppt * ppf).squeeze(0)

            # peak picking
            cfp = torch.where((cfp > cfp.roll(1, 1)) & (cfp > cfp.roll(-1, 1)), cfp, 0)
            cfp = cfp[:, selection_slice]
            stream1, stream2 = get_streams(
                cfp, np.arange(len(freqs)), n=2, thresh_ratio=args.thresh
            )

            singer1_H = torch.zeros(cfp.shape[0], len(freqs) + 1)
            singer2_H = singer1_H.clone()
            singer1_H.scatter_(
                1,
                torch.tensor(
                    list(
                        map(
                            lambda x: [len(freqs)] * 3
                            if x is None
                            else [x - 1, x, x + 1],
                            stream1,
                        )
                    )
                ),
                1,
            )
            singer2_H.scatter_(
                1,
                torch.tensor(
                    list(
                        map(
                            lambda x: [len(freqs)] * 3
                            if x is None
                            else [x - 1, x, x + 1],
                            stream2,
                        )
                    )
                ),
                1,
            )

            nmf = NMF(
                W=torch.cat([W_f0, W_f0], dim=1),
                H=torch.cat([singer1_H, singer2_H], dim=1),
                trainable_W=False,
            )

            X = spec(x)

            nmf.fit(X.abs().T, beta=args.beta, alpha=1e-6)

            with torch.no_grad():
                H1, H2 = nmf.H.chunk(2, dim=1)
                W1, W2 = nmf.W.chunk(2, dim=1)
                recon_singer1 = H1 @ W1.T
                recon_singer2 = H2 @ W2.T
                recon = recon_singer1 + recon_singer2
                mask1 = recon_singer1 / recon
                mask2 = recon_singer2 / recon

            y1 = inv_spec(X * mask1.T)
            y2 = inv_spec(X * mask2.T)

            result = torch.stack([y1, y2], dim=0)

            if result.shape[1] < y.shape[1]:
                result = F.pad(
                    result.unsqueeze(0),
                    (0, y.shape[1] - result.shape[1]),
                ).squeeze(0)

            loss, reordered_sources = loss_func(
                result.unsqueeze(0), y.unsqueeze(0), return_est=True
            )
            est = reordered_sources.squeeze().cpu().numpy()

            utt_metrics = get_metrics(
                x.numpy(),
                y.numpy(),
                est,
                sample_rate=sr,
                metrics_list=COMPUTE_METRICS,
            )

            # calculate improvement
            for metric in COMPUTE_METRICS:
                v = utt_metrics.pop("input_" + metric)
                utt_metrics[metric + "i"] = utt_metrics[metric] - v

            for k, v in utt_metrics.items():
                if k not in accumulate_metrics_mean:
                    accumulate_metrics_mean[k] = 0

                accumulate_metrics_mean[k] += (v - accumulate_metrics_mean[k]) / (
                    mix_num + 1
                )

            pbar.set_postfix(accumulate_metrics_mean)

            if args.out is not None:
                out_dir = pathlib.Path(args.out) / f"medleyvox_{mix_num}"
                out_dir.mkdir(parents=True, exist_ok=True)

                sf.write(
                    out_dir / "mixture.wav",
                    x.numpy(),
                    sr,
                    "PCM_16",
                )

                for i, s in enumerate(est):
                    out_path = out_dir / f"{ids[i]}.wav"
                    sf.write(out_path, s, sr, "PCM_16")

    print(accumulate_metrics_mean)
