import argparse
import hydra
import os
import torch
import pathlib
from tqdm import tqdm
from typing import Callable, Optional
from math import sqrt
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import soundfile as sf

from medley_vox import MedleyVox

COMPUTE_METRICS = ["si_sdr", "sdr"]


@torch.no_grad()
def separate_mixture(
    mixture: torch.Tensor,
    noises: torch.Tensor,
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    cond_index: int = 0,
    s_churn: float = 40.0,  # > 0 to add randomness
    num_resamples: int = 2,
    use_tqdm: bool = False,
    gaussian: bool = False,
):
    # Set initial noise
    x = sigmas[0] * noises  # [batch_size, num-sources, sample-length]

    for i in tqdm(range(len(sigmas) - 1), disable=not use_tqdm):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        for r in range(num_resamples):
            # Inject randomness
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            sigma_hat = sigma * (gamma + 1)
            x = x + torch.randn_like(x) * (sigma_hat**2 - sigma**2) ** 0.5

            if cond is not None:
                noisey_cond = cond + torch.randn_like(cond) * sigma
                x[:, :cond_index] = noisey_cond

            # Compute conditioned derivative
            if not gaussian:
                x[:1] = mixture - x[1:].sum(dim=0, keepdim=True)
            score = (x - denoise_fn(x, sigma=sigma)) / sigma
            if gaussian:
                d = score + sigma / (2 * gamma**2) * (mixture - x.sum(dim=0))
                x += d * (sigma_next - sigma_hat)
            else:
                ds = score[1:] - score[:1]

                # Update integral
                x[1:] += ds * (sigma_next - sigma_hat)

            # Renoise if not last resample step
            if r < num_resamples - 1:
                x = x + torch.sqrt(sigma**2 - sigma_next**2) * torch.randn_like(x)

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("medleyvox", type=str, help="Path to MedleyVox dataset")
    parser.add_argument("-T", default=100, type=int, help="Number of diffusion steps")
    parser.add_argument("--out", type=str, help="Output directory")
    parser.add_argument("--cond", action="store_true", help="Use conditioning")
    parser.add_argument("--hop-length", type=int, help="Hop length")

    args = parser.parse_args()

    config_path, config_name = os.path.split(args.config)

    with hydra.initialize(config_path=config_path):
        cfg = hydra.compose(config_name=config_name)

    sr = cfg.sampling_rate
    length = cfg.length

    model = hydra.utils.instantiate(cfg.model)
    vctk_checkpoint = torch.load(
        args.ckpt,
        map_location="cpu",
    )
    model.load_state_dict(vctk_checkpoint["state_dict"])
    diffusion_schedule = hydra.utils.instantiate(
        cfg.callbacks.audio_samples_logger.diffusion_schedule
    )

    model = model.cuda()
    model.eval()
    diffusion_schedule = diffusion_schedule.cuda()

    inner_denoise_fn = model.model.diffusion.diffusion.denoise_fn

    def denoise_fn(x, sigma):
        x = x.unsqueeze(1)
        return inner_denoise_fn(x, sigma=sigma).squeeze(1)

    sigmas = diffusion_schedule(args.T, "cuda")

    dataset = MedleyVox(
        args.medleyvox,
        sample_rate=sr,
    )

    hop_length = length // 2 if args.hop_length is None else args.hop_length
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    print(length / sr, hop_length / sr)

    accumulate_metrics_mean = {}

    with tqdm(dataset) as pbar:
        for mix_num, (x_cpu, y_cpu, ids) in enumerate(pbar):
            x = x_cpu.cuda()
            y = y_cpu.cuda()
            n = y.shape[0]

            if args.cond:
                cond = torch.zeros(n, length).cuda()
                sub_m = torch.zeros(length).cuda()
                result = []
                for sub_x, sub_y in zip(
                    torch.split(x, hop_length), torch.split(y, hop_length, 1)
                ):
                    noise = torch.randn(n, length).cuda()
                    overlap_size = length - sub_x.numel()
                    sub_m = torch.cat([sub_m[-overlap_size:], sub_x])
                    cond = cond[:, -overlap_size:]
                    pred = separate_mixture(
                        sub_m,
                        noise,
                        denoise_fn,
                        sigmas,
                        s_churn=0.0,
                        cond=cond,
                        cond_index=overlap_size,
                        use_tqdm=False,
                        gaussian=False,
                    )
                    cond = torch.cat([cond, sub_y], dim=1)
                    result.append(pred[:, -sub_x.numel() :])

                    if len(result) == 1:
                        loss, pred = loss_func(
                            result[0].unsqueeze(0), sub_y.unsqueeze(0), return_est=True
                        )
                        result[0] = pred.squeeze()

                result = torch.cat(result, dim=1)
            else:
                original_length = x.numel()
                padding = length - (original_length % length)
                if padding < length:
                    x = torch.cat([x, x.new_zeros(padding)], dim=0)

                cond = None
                result = torch.empty(n, x.numel()).cuda()
                for i in range(0, x.numel() - hop_length, hop_length):
                    chunk = x[i : i + length]
                    noise = torch.randn(n, length).cuda()
                    pred = separate_mixture(
                        chunk,
                        noise,
                        denoise_fn,
                        sigmas,
                        s_churn=0.0,
                        cond=cond,
                        cond_index=hop_length,
                        use_tqdm=False,
                    )
                    cond = pred[:, hop_length:]
                    result[:, i : i + length] = pred

                result = result[:, :original_length]

            loss, reordered_sources = loss_func(
                result.unsqueeze(0), y.unsqueeze(0), return_est=True
            )
            est = reordered_sources.squeeze().cpu().numpy()

            utt_metrics = get_metrics(
                x_cpu.numpy(),
                y_cpu.numpy(),
                est,
                sample_rate=sr,
                metrics_list=COMPUTE_METRICS,
            )

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
                    x_cpu.numpy(),
                    sr,
                    "PCM_16",
                )

                for i, s in enumerate(est):
                    out_path = out_dir / f"{ids[i]}.wav"
                    sf.write(out_path, s, sr, "PCM_16")
