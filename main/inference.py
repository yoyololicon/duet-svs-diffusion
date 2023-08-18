import abc
from functools import partial
from pathlib import Path
from typing import List, Optional, Callable, Mapping
import pytorch_lightning as pl

import torch
import torchaudio
import tqdm
from math import sqrt, ceil
import hydra
from audio_data_pytorch.utils import fractional_random_split

from audio_diffusion_pytorch.diffusion import Schedule
from torch.utils.data import DataLoader
from torchaudio.datasets import VCTK_092

from main.dataset import vctk_collate
from main.module_vckt import Model


class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def separate(mixture, num_steps) -> Mapping[str, torch.Tensor]:
        ...


class WeaklyMSDMSeparator(Separator):
    def __init__(self, stem_to_model: Mapping[str, Model], sigma_schedule, **kwargs):
        super().__init__()
        self.stem_to_model = stem_to_model
        self.separation_kwargs = kwargs
        self.sigma_schedule = sigma_schedule

    def separate(self, mixture: torch.Tensor, num_steps: int):
        stems = self.stem_to_model.keys()
        models = [self.stem_to_model[s] for s in stems]
        fns = [m.model.diffusion.diffusion.denoise_fn for m in models]

        # get device of models
        devices = {m.device for m in models}
        assert len(devices) == 1, devices
        (device,) = devices

        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape

        def denoise_fn(x, sigma):
            xs = [x[:, i:i + 1] for i in range(4)]
            xs = [fn(x, sigma=sigma) for fn, x in zip(fns, xs)]
            return torch.cat(xs, dim=1)

        y = separate_mixture(
            mixture=mixture,
            denoise_fn=denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(stems), length_samples).to(device),
            **self.separation_kwargs,
        )
        return {stem: y[:, i:i + 1, :] for i, stem in enumerate(stems)}


# Algorithms ------------------------------------------------------------------

def differential_with_dirac(x, sigma, denoise_fn, mixture, source_id=0):
    num_sources = x.shape[1]
    x[:, [source_id], :] = mixture - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
    score = (x - denoise_fn(x, sigma=sigma)) / sigma
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s - score[:, source_id] for s in scores]
    return torch.stack(ds, dim=1)


def differential_with_gaussian(x, sigma, denoise_fn, mixture, gamma_fn=None):
    gamma = sigma if gamma_fn is None else gamma_fn(sigma)
    d = (x - denoise_fn(x, sigma=sigma)) / sigma
    d = d - sigma / (2* gamma ** 2) * (mixture - x.sum(dim=[1], keepdim=True))
    return d


@torch.no_grad()
def separate_mixture(
        mixture: torch.Tensor,
        denoise_fn: Callable,
        sigmas: torch.Tensor,
        noises: Optional[torch.Tensor],
        differential_fn: Callable = differential_with_dirac,
        s_churn: float = 20.0,  # > 0 to add randomness
        num_resamples: int = 2,
        use_tqdm: bool = False,
):
    # Set initial noise
    x = sigmas[0] * noises  # [batch_size, num-sources, sample-length]

    vis_wrapper = tqdm.tqdm if use_tqdm else lambda x: x
    for i in vis_wrapper(range(len(sigmas) - 1)):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        for r in range(num_resamples):
            # Inject randomness
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
            sigma_hat = sigma * (gamma + 1)
            x = x + torch.randn_like(x) * (sigma_hat ** 2 - sigma ** 2) ** 0.5

            # Compute conditioned derivative
            d = differential_fn(mixture=mixture, x=x, sigma=sigma_hat, denoise_fn=denoise_fn)

            # Update integral
            x = x + d * (sigma_next - sigma_hat)

            # Renoise if not last resample step
            if r < num_resamples - 1:
                x = x + sqrt(sigma ** 2 - sigma_next ** 2) * torch.randn_like(x)

    return x.cpu().detach()


# -----------------------------------------------------------------------------
def save_separation(
        separated_tracks: Mapping[str, torch.Tensor],
        sample_rate: int,
        chunk_path: Path,
):
    for stem, separated_track in separated_tracks.items():
        torchaudio.save(chunk_path / f"{stem}.wav", separated_track.cpu(), sample_rate=sample_rate)


if __name__ == '__main__':

    pl.seed_everything(12345)
    import torchmetrics

    with hydra.initialize(config_path=".."):
        cfg = hydra.compose(config_name="exp/base_vctk_k_none.yaml")

    model = hydra.utils.instantiate(cfg['model']).cuda()
    vctk_checkpoint = torch.load('/home/emilian/PycharmProjects/multi-speaker-diff-sep/data/epoch=117-valid_loss=0.015.ckpt',
                                  map_location='cuda')
    model.load_state_dict(vctk_checkpoint['state_dict'])
    diffusion_schedule = hydra.utils.instantiate(cfg['callbacks']['audio_samples_logger']['diffusion_schedule']).cuda()
    separator = WeaklyMSDMSeparator(stem_to_model = {"separated_0": model.cuda(),
                                                     "separated_1": model.cuda()},
                                    sigma_schedule=diffusion_schedule,
                                    use_tqdm=True)

    dataset = VCTK_092(root="/home/emilian/PycharmProjects/multi-speaker-diff-sep/data/vctk", download=True)
    split = [1.0 - 0.01, 0.01]
    _, data_val = fractional_random_split(dataset, split)
    dataloader = DataLoader(dataset=data_val,
                            batch_size=2,
                            num_workers=0,
                            pin_memory=False,
                            drop_last=True,
                            shuffle=False,
                            collate_fn=partial(vctk_collate, mix_k=None))
    data_iter = iter(dataloader)
    _ = next(data_iter)
    _ = next(data_iter)
    _ = next(data_iter)
    _ = next(data_iter)
    _ = next(data_iter)
    batch = next(data_iter)

    mix = batch.sum(dim=0, keepdim=True)
    torchaudio.save("./source_0.wav", batch[0].cpu(), sample_rate=22050)
    torchaudio.save("./source_1.wav", batch[1].cpu(), sample_rate=22050)
    separated_tracks = separator.separate(mixture=mix, num_steps=500)
    save_separation({k: v[0] for k, v in separated_tracks.items()},
                    sample_rate=22050,
                    chunk_path=Path('.'))
    torchaudio.save("./mix.wav", mix[0].cpu(), sample_rate=22050)