import functools
from pathlib import Path
from typing import Any, Tuple

import torch
from hydra.core.config_store import ConfigStore

from diba import Likelihood, SeparationPrior
from transformers import GPT2LMHeadModel


class UnconditionedTransformerPrior(SeparationPrior):
    def __init__(self, transformer: GPT2LMHeadModel, sos: int):
        self.transformer = transformer
        self.transformer.eval()
        self.label = sos

    def get_sos(self):
        return torch.tensor(self.label, dtype=torch.long).view(1).to(self.get_device())

    def get_device(self) -> torch.device:
        return self.transformer.device

    def get_tokens_count(self) -> int:
        return self.transformer.lm_head.out_features

    def get_logits(self, token_ids: torch.LongTensor, past_key_values: Any = None) -> Tuple[torch.Tensor, Any]:
        token_ids = token_ids[:, -1:] if past_key_values is not None else token_ids
        output = self.transformer(token_ids, past_key_values=past_key_values)
        logits, past = output.logits, output.past_key_values
        return logits[:, -1, :], past

    def reorder_cache(self, past_key_values: Any, beam_idx) -> Any:
        return self.transformer._reorder_cache(past_key_values, beam_idx)


class DenseLikelihood(Likelihood):
    def __init__(self, sums: torch.Tensor):
        # normalize the sums over the conditionals in order to get the likelihood function at each z_m
        normalization = torch.sum(sums, dim=-1)
        mask = normalization != 0.0
        sums[mask] = sums[mask] / normalization[mask].unsqueeze(-1)
        self.sum = sums

    def get_device(self) -> torch.device:
        return self.sum.device

    def get_tokens_count(self) -> int:
        return self.sum.shape[0]

    @functools.lru_cache(512)
    def get_log_likelihood(self, token_idx: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        mixture_slice = self.sum[:, :, token_idx]
        coords = torch.nonzero(mixture_slice).transpose(0, 1)
        return coords, torch.log(mixture_slice[coords[0], coords[1]])
