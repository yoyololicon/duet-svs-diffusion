from pathlib import Path
from typing import Union, Tuple, Optional, Sequence, Any

import av
import numpy as np
import torch
import torchaudio

from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae, make_prior

from jukebox.vqvae.vqvae import calculate_strides, VQVAE


SCRIPT_DIRECTORY = Path(__file__).parent
ROOT_DIRECTORY = SCRIPT_DIRECTORY.parent


def load_audio(
    filename: Union[str, Path],
    sample_rate: Optional[int] = None,
    audio_layout: str = "stereo",
) -> Tuple[np.ndarray, int]:
    container = av.open(str(filename))
    audio = container.streams.get(audio=0)[0]  # Only first audio stream

    file_sample_rate = audio.time_base.denominator
    sample_rate = file_sample_rate if sample_rate is None else sample_rate
    assert audio.time_base.numerator == 1

    audio_layout = av.AudioLayout(audio_layout)
    resampler = av.AudioResampler(format="fltp", layout=audio_layout, rate=sample_rate)
    sig = np.zeros(
        (
            len(audio_layout.channels),
            int(audio.duration * (sample_rate / file_sample_rate)) + 100,
        ),
        dtype=np.float32,
    )

    total_read = 0
    for frame in container.decode(audio=0):  # Only first audio stream
        frame_data = resampler.resample(frame).to_ndarray()
        read = frame_data.shape[-1]
        sig[:, total_read : total_read + read] = frame_data
        total_read += read

    sig = sig[:total_read]
    return sig, sample_rate


def save_audio(
    signal: torch.Tensor, filename: Union[str, Path], sample_rate: int,
):
    assert len(signal.shape) == 2
    torchaudio.save(src=signal, filepath=str(filename), sample_rate=sample_rate)


def is_silent(signal: torch.Tensor, silence_threshold: float = 1.5e-5) -> bool:
    assert_is_audio(signal)
    num_samples = signal.shape[-1]
    return torch.linalg.norm(signal) / num_samples < silence_threshold


def get_nonsilent_chunks(
    track_1: torch.Tensor,
    track_2: torch.Tensor,
    max_chunk_size: int,
    min_chunk_size: int = 0,
):
    assert_is_audio(track_1, track_2)
    _, track_1_samples = track_1.shape
    _, track_2_samples = track_2.shape
    assert track_1_samples == track_2_samples

    num_samples = min(track_1_samples, track_2_samples)
    num_chunks = num_samples // max_chunk_size + int(num_samples % max_chunk_size != 0)

    available_chunks = []
    for i in range(num_chunks):
        m1 = track_1[:, i * max_chunk_size : (i + 1) * max_chunk_size]
        m2 = track_2[:, i * max_chunk_size : (i + 1) * max_chunk_size]

        _, m1_samples = m1.shape
        _, m2_samples = m2.shape

        if (
            not (is_silent(m1) or is_silent(m2))
            and m1_samples >= min_chunk_size
            and m2_samples >= min_chunk_size
        ):
            available_chunks.append(i)

    return available_chunks


def load_audio_tracks(
    path_1: Union[str, Path], path_2: Union[str, Path], sample_rate: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    signal_0, sr_0 = load_audio(path_1, sample_rate=sample_rate, audio_layout="mono")
    signal_1, sr_1 = load_audio(path_2, sample_rate=sample_rate, audio_layout="mono")
    signal_0 = torch.from_numpy(signal_0)
    signal_1 = torch.from_numpy(signal_1)
    assert sr_0 == sr_1 == sample_rate
    return signal_0, signal_1


def assert_is_audio(*signal: Union[torch.Tensor, np.ndarray]):
    for s in signal:
        assert len(s.shape) == 2
        assert s.shape[0] == 1


def setup_vqvae(
    vqvae_path: Union[str, Path],
    vqvae_type: str,
    sample_tokens: int,
    sample_rate: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # construct vqvae
    hps = setup_hparams(vqvae_type, dict(sr=sample_rate, restore_vqvae=str(vqvae_path)))
    raw_to_tokens = get_raw_to_tokens(hps.strides_t, hps.downs_t)
    hps.sample_length = sample_tokens * raw_to_tokens
    return make_vqvae(hps, device)


def setup_priors(
    prior_paths: Sequence[Union[str, Path]],
    prior_types: Sequence[str],
    vqvae: VQVAE,
    fp16: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # construct openai priors
    priors = []
    for pidx, prior_path in enumerate(prior_paths):
        priors.append(
            make_prior(
                setup_hparams(
                    prior_types[pidx],
                    dict(
                        level=vqvae.levels - 1,
                        labels=None,
                        restore_prior=str(prior_path),
                        c_res=1,
                        fp16_params=fp16,
                        n_ctx=8192,
                    ),
                ),
                vqvae,
                device,
            )
        )
    return priors


def get_raw_to_tokens(
    strides_t: Sequence[int], downs_t: Sequence[int], level: Optional[int] = None
) -> int:
    assert len(strides_t) == len(downs_t)
    level = len(strides_t) - 1 if level is None else level
    return np.prod(calculate_strides(strides_t, downs_t)[: level + 1])


def decode_latents(vqvae: VQVAE, z: torch.Tensor, level:int):
    x = vqvae.decoders[level]([z], all_levels=False)
    return vqvae.postprocess(x)


def decode_latent_codes(vqvae: VQVAE, zq: torch.LongTensor, level: Optional[int] = None):
    assert len(zq.shape) == 1
    n_tokens, = zq.shape

    level = vqvae.levels - 1 if level is None else level
    rtt = get_raw_to_tokens(vqvae.strides_t, vqvae.downs_t, level)
    rec = vqvae.decode([zq.view(1, n_tokens)], start_level=level)
    return rec.view(n_tokens*rtt)


def get_dataset_subsample(data_length: int, num_samples: Optional[int] = None, seed: int = 0) -> Sequence[int]:
    num_samples = num_samples if num_samples is not None else data_length
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(data_length, generator=generator)[:num_samples].tolist()