import torch
from torch import nn
import torch.nn.functional as F
from audio_diffusion_pytorch import UNetV0
from typing import List
from functools import reduce

from .model import DiffusionEmbedding


class Mousai(nn.Module):
    def __init__(self, factors: List[int], **kwargs) -> None:
        super().__init__()

        self.unet = UNetV0(dim=1, in_channels=1, factors=factors, **kwargs)
        self.minimum_common_divisor = reduce(lambda x, y: x * y, factors)

    def forward(self, audio, diffusion_step, _=None):
        audio = audio.unsqueeze(1)
        T = audio.shape[2]
        if T % self.minimum_common_divisor != 0:
            audio = F.pad(
                audio,
                pad=[
                    0,
                    self.minimum_common_divisor - T % self.minimum_common_divisor,
                ],
                mode="reflect",
                value=0,
            )
        y = self.unet(audio, diffusion_step)[..., :T]
        # if y.shape[2] < audio.shape[1]:
        #     y = torch.cat([y, audio[:, None, y.shape[2] :]], dim=2)
        return y.squeeze(1)
