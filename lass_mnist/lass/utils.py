from pathlib import Path
from typing import Tuple

import torch
from hydra.core.config_store import ConfigStore

from modules import VectorQuantizedVAE

# useful paths
ROOT_DIR = Path(__file__).parent.parent
CONFIG_DIR = ROOT_DIR / "configs"

# singleton objects
CONFIG_STORE = ConfigStore.instance()


def refine_latents(
    model: VectorQuantizedVAE,
    latents_1: torch.Tensor,
    latents_2: torch.Tensor,
    mixtures: torch.Tensor,
    n_iterations: int = 2000,
    learning_rate: float = 1e-3,
    regularizer_coeff: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refine latent codes for better separation.

    The refinment operation uses Adam to optimize the latent vectors in input
    so that their mixture is

    Args:
        model: vq-vae model
        latents_1: separation latent codes. Shape (batch size, ...)
        latents_2: separation latent codes. Shape (batch size, ...)
        mixtures: mixtures. (batch size, ...)
        n_iterations: number of iterations of the optimization procedure
        learning_rate: learning rate to iuse
        regularizer_coeff: coefficient used for regularization

    Returns:
        The two separated signals.
    """
    # DO NOT REMOVE: necessary, although not sure why
    latents_1 = torch.stack([latents_1], 0).squeeze(0)
    latents_2 = torch.stack([latents_2], 0).squeeze(0)

    # copy initial values
    gen1 = latents_1.clone().detach().requires_grad_(True)
    gen2 = latents_2.clone().detach().requires_grad_(True)
    latents_1 = latents_1.clone().detach()
    latents_2 = latents_2.clone().detach()
    mixtures = mixtures.clone().detach()

    optimizer = torch.optim.Adam([gen1, gen2], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)

    # optimize
    for s in range(n_iterations):
        geni1 = model.decode_latents(gen1)
        geni2 = model.decode_latents(gen2)
        geni_mixtures = (geni1 + geni2) / 2.0

        reg = regularizer_coeff * torch.mean((gen1 - latents_1) ** 2 + (gen2 - latents_2) ** 2)
        loss = torch.mean((geni_mixtures - mixtures).pow(2)) + reg
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return geni1, geni2
