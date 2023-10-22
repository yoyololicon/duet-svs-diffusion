import functools
from typing import Any, Optional, Tuple
import numpy as np
import sparse
from diba.diba import Likelihood, SeparationPrior
import torch

from jukebox.prior.autoregressive import ConditionalAutoregressive2D


class JukeboxPrior(SeparationPrior):
    def __init__(self, transformer: ConditionalAutoregressive2D, x_cond: Optional[torch.Tensor] = None):
        self._prior = transformer
        self._x_cond = x_cond

    @functools.lru_cache(1)
    def get_device(self) -> torch.device:
        return list(self._prior.transformer.parameters())[0].device

    def get_tokens_count(self) -> int:
        return self._prior.x_out.out_features

    def get_sos(self) -> Any:
        return 0 #self._prior.start_token # DUMMY METHOD

    def get_logits(
            self, token_ids: torch.LongTensor, cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:

        # get dimensions
        n_samples, seq_length = token_ids.shape
        sample_t = seq_length - 1

        # delete cache if token_ids is 1 (sos token)
        assert len(token_ids) > 0
        if len(token_ids) == 1:
            self._prior.transformer.del_cache()

        x = token_ids[:, -1:]

        # add x-conditioning
        if self._x_cond is not None:
            x_cond = self._x_cond
        else:
            x_cond = torch.zeros((n_samples, 1, -1), device=x.device, dtype=torch.float)

        # get embeddings
        x, cond_0 = self._prior.get_emb(sample_t, n_samples, x, x_cond=x_cond, y_cond=None)

        self._prior.transformer.check_cache(n_samples, sample_t, fp16=True)
        x = self._prior.transformer(x, sample=True, fp16=True) # TODO: try sample = False
        x = self._prior.x_out(x)[:,-1,:]
        return x.to(torch.float32), None

    def reorder_cache(self, cache: Any, beam_idx: torch.LongTensor) -> Any:
        self._prior.transformer.substitute_cache(beam_idx)
        return None


class SparseLikelihood(Likelihood):
    def __init__(self, sum_dist_path: str, device: torch.device, lambda_coeff: float = 1.0):
        self._device = torch.device(device)
        self._lambda_coeff = lambda_coeff
        self._freqs = self._normalize_matrix(sum_dist_path)


    def get_device(self) -> torch.device:
        return self._device

    def get_tokens_count(self) -> int:
        return self._freqs.shape[0]

    @functools.lru_cache(512)
    def get_log_likelihood(self, token_idx: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        sparse_nll = self._freqs[:, :, token_idx].tocoo()
        nll_coords = torch.tensor(sparse_nll.coords, device=self.get_device(), dtype=torch.long)
        nll_data = torch.tensor(sparse_nll.data, device=self.get_device(), dtype=torch.float)
        return nll_coords, nll_data

    def _normalize_matrix(self, sum_dist_path: str):
        sum_dist = sparse.load_npz(str(sum_dist_path))
        integrals = sum_dist.sum(axis=-1, keepdims=True)
        I, J, _ = integrals.coords
        integrals = sparse.COO(
            integrals.coords, integrals.data, shape=integrals.shape, fill_value=1
        )
        log_data = np.log(sum_dist / integrals) * self._lambda_coeff
        return sparse.GCXS.from_coo(log_data, compressed_axes=[2])

