"""Main module."""
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy
import numpy as np
import torch
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from diba.interfaces import Likelihood, SeparationPrior
from diba.utils import get_topk, normalize_logits

# def _print_beams(xs_0, xs_1, scores, posterior_data, ll_coords):
#
#     def _print_next_tokens(bi, xs, pi):
#         prefix_score = scores[bi].item()
#         path = '->'.join(str(bi) for bi in xs[bi].tolist())
#         print(f"[{bi}]  p{pi} {path} = {prefix_score}")
#         for j, coords in enumerate(ll_coords.T):
#             next_token = coords[pi]
#             print(f"      * {path}->{next_token} = {prefix_score:.2} + {posterior_data[bi,j]:.2}")
#
#     for i in range(len(xs_0)):
#         _print_next_tokens(i, xs_0, 0)
#         _print_next_tokens(i, xs_1, 1)
#         print()
#     print("======================")
#
#
# def _print_logits(logits, x_0, x_1, num_tokens):
#     num_samples, _ = logits.shape
#     for b, c0, c1 in torch.nonzero(logits.view(num_samples, num_tokens, num_tokens) != -torch.inf):
#         path_0 = '->'.join(str(bi) for bi in x_0[b].tolist())
#         path_1 = '->'.join(str(bi) for bi in x_1[b].tolist())
#         print(f"[{c0 * num_tokens + c1}]: {path_0}->{c0}, {path_1}->{c1} ({logits[b, c0 * num_tokens + c1]})")
#     print("----------------")


class _SeparationModel(PreTrainedModel):
    def __init__(
        self,
        prior_0: SeparationPrior,
        prior_1: SeparationPrior,
        likelihood: Likelihood,
        mixture: Sequence[int],
        temperature: float = 1.0,
    ):
        super().__init__(PretrainedConfig())
        self.prior_0, self.prior_1 = prior_0, prior_1
        self.likelihood = likelihood
        self.mixture = mixture
        self.temperature = temperature
        self._dummy_tensor = torch.zeros(
            1, device=self.prior_0.get_device()
        )  # IMPORTANT! used by hugging face to infer device
        self.sample_t = 0

    def _reset(self):
        self.sample_t = 0

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """Prepare inputs for generation."""
        return {"input_ids": input_ids, **kwargs}

    def _reorder_cache(self, past: Tuple[Any, Any], beam_idx: torch.LongTensor):
        past_0, past_1 = past
        return (
            self.prior_0._reorder_cache(past_0, beam_idx),
            self.prior_1._reorder_cache(past_1, beam_idx),
        )

    def get_bos(self) -> int:
        return self.prior_0.get_sos() * self.likelihood.get_tokens_count() + self.prior_1.get_sos()

    def forward(
        self,
        input_ids: torch.LongTensor,
        past: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ):
        torch.set_printoptions(precision=2, sci_mode=False)
        """Forward method."""
        num_tokens = self.likelihood.get_tokens_count()
        device = self.likelihood.get_device()
        devices = {self.likelihood.get_device(), self.prior_0.get_device(), self.prior_1.get_device()}
        if len(devices) > 1:
            raise RuntimeError(
                "Error: inconsistent devices detected, likelihood and priors are not on the "
                f"same device (devices: {devices})."
            )

        x_0 = input_ids.div(num_tokens, rounding_mode="trunc")
        x_1 = input_ids % num_tokens

        past_0, past_1 = (None, None) if past is None else past

        # compute log priors
        log_p_0, past_0 = self.prior_0._get_logits(x_0, past_key_value=past_0)
        log_p_1, past_1 = self.prior_1._get_logits(x_1, past_key_value=past_1)

        # NOTE: during first token separation batch-size should be 1
        assert len(log_p_0) == len(log_p_1)
        assert log_p_0.shape[-1] == log_p_1.shape[-1] == num_tokens

        # normalize priors and apply temperature
        log_p_0 = normalize_logits(log_p_0, self.temperature)
        log_p_1 = normalize_logits(log_p_1, self.temperature)

        # log likelihood in sparse COO format
        assert isinstance(self.mixture[self.sample_t], int)
        ll_coords, ll_data = self.likelihood._get_log_likelihood(self.mixture[self.sample_t])

        # compute log posterior
        if ll_coords.numel() == 0:
            raise RuntimeError(f"Code {self.mixture[self.sample_t]} is not available in likelihood!")

        # Note: posterior_data has shape (num_samples, nonzeros)
        posterior_data = _compute_log_posterior(ll_data, ll_coords, log_p_0, log_p_1)

        # Convert to shape (num samples, num_tokens*num_tokens)
        coords0, coords1 = ll_coords
        num_samples, _ = posterior_data.shape
        logits = torch.full(size=(num_samples, num_tokens**2), fill_value=numpy.log(1e-16), device=device)
        logits.index_copy_(-1, coords0 * num_tokens + coords1, posterior_data)

        # update variables
        self.sample_t += 1

        return CausalLMOutputWithPast(
            logits=logits.view(num_samples, 1, num_tokens**2),
            past_key_values=None if past_0 is None and past_1 is None else (past_0, past_1),
        )


def separate(
    priors: Sequence[SeparationPrior],
    likelihood: Likelihood,
    mixture: Sequence[int],
    temperature: float = 1.0,
    do_sample: Optional[bool] = None,
    num_beams: Optional[int] = None,
    num_beam_groups: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    num_return_sequences: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Separate input mixtures.

    Args:
        priors: separation priors
        likelihood: likelihood of tokens mix
        mixture: mixture to separate
        temperature: temperture used by the priors
        do_sample: whether to use sampling or beam search
        num_beams: number of beams to use during beam search
        num_beam_groups: number of groups to use during diverse beam search
        top_k: top-k value to use during sampling
        top_p: top-p value to use during sampling
        num_return_sequences: number of sequence to return
        **kwargs: additional parameters

    Returns:
        (s0, s1) returned separated signals. They have shape (num. return sequences, mixture length)
    """
    p_0, p_1 = priors
    sep_model = _SeparationModel(p_0, p_1, likelihood, mixture=mixture, temperature=temperature)
    result = sep_model.generate(
        bos_token_id=sep_model.get_bos(),
        max_length=len(mixture) + 1,
        min_length=len(mixture) + 1,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=do_sample,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        **kwargs,
    )

    num_tokens = likelihood.get_tokens_count()
    result = result.sequences[:, 1:]
    r0 = result.div(num_tokens, rounding_mode="trunc")
    r1 = result % num_tokens
    return r0, r1


def _compute_log_posterior(
    nll_data: torch.Tensor,
    nll_coords: torch.LongTensor,
    log_p0: torch.Tensor,
    log_p1: torch.Tensor,
):
    coords_p0, coords_p1 = nll_coords
    return nll_data + log_p0[:, coords_p0] + log_p1[:, coords_p1]


def _ancestral_sample(
    priors: Sequence[SeparationPrior],
    likelihood: Likelihood,
    mixture: Sequence[int],
    n_samples: int,
    temperature: float,
):
    prior_0, prior_1 = priors
    sample_tokens = len(mixture)

    # check that everything is on the same device
    assert len(set([p._get_device() for p in priors] + [likelihood.get_device()])) == 1
    device = likelihood.get_device()

    # initialize loop variables
    log_post_sum = torch.zeros(1, 1, dtype=torch.long, device=device)
    xs_0, xs_1 = torch.full((2, n_samples, sample_tokens + 1), fill_value=-1, dtype=torch.long, device=device)
    xs_0[:, 0], xs_1[:, 0] = [p.get_sos() for p in priors]
    past_0, past_1 = (None, None)
    num_current_beams = 1

    # loop over all samples tokens
    for sample_t in tqdm(range(sample_tokens)):

        # compute log priors
        log_p_0, past_0 = prior_0._get_logits(xs_0[:num_current_beams, : sample_t + 1], past_0)
        log_p_1, past_1 = prior_1._get_logits(xs_1[:num_current_beams, : sample_t + 1], past_1)

        # NOTE: during first token separation batch-size should be 1
        assert len(log_p_0) == len(log_p_1)
        assert len(log_p_0) == 1 if sample_t == 0 else len(log_p_0) <= n_samples
        assert log_p_0.shape[-1] == log_p_1.shape[-1] == likelihood.get_tokens_count()

        # normalize priors and apply temperature
        log_p_0 = normalize_logits(log_p_0, temperature)
        log_p_1 = normalize_logits(log_p_1, temperature)

        # log likelihood in sparse COO format
        assert isinstance(mixture[sample_t], int)
        ll_coords, ll_data = likelihood._get_log_likelihood(mixture[sample_t])

        # compute log posterior
        if ll_coords.numel() > 0:
            # Note: posterior_data has shape (n_samples, nonzeros)
            posterior_data = _compute_log_posterior(ll_data, ll_coords, log_p_0, log_p_1)
            log_post_sum, (beams, coords_idx) = get_topk(log_post_sum + posterior_data, n_samples)
            log_post_sum = log_post_sum.unsqueeze(-1)
            x_0, x_1 = ll_coords[:, coords_idx]
        else:
            raise RuntimeError(f"Code {mixture[sample_t]} is not available in likelihood!")

        num_current_beams = len(beams)

        # make history consistent with current beams
        xs_0[:num_current_beams, : sample_t + 1] = xs_0[beams, : sample_t + 1]
        xs_1[:num_current_beams, : sample_t + 1] = xs_1[beams, : sample_t + 1]

        # Note: x_0, x_1 have shape (n_sample,)
        xs_0[:, sample_t + 1] = x_0
        xs_1[:, sample_t + 1] = x_1

        log_post_sum = log_post_sum[beams]

        # update the priors cache w.r.t. the current beams
        past_0 = prior_0._reorder_cache(past_0, beams)
        past_1 = prior_1._reorder_cache(past_1, beams)

    result_0, result_1 = xs_0[:num_current_beams, 1:], xs_1[:num_current_beams, 1:]

    # check returned separation is correctly masked
    assert (result_0 == -1).sum() == 0
    assert (result_1 == -1).sum() == 0

    return result_0, result_1


@torch.no_grad()
def fast_beamsearch_separation(
    priors: Sequence[SeparationPrior],
    likelihood: Likelihood,
    mixture: Sequence[int],
    num_beams: int,
    temperature: float = 1.0,
) -> Tuple[Sequence[int], Sequence[int]]:
    """Separate the input mixture using beam-search.

    Args:
        priors: List of priors to use for separation
        likelihood: Likelihood to use for separation
        mixture: Sequence of tokens to separate
        num_beams: number of beams to use for the beamsearch
        temperature: temperature of the priors

    Returns:
        (s0, s1) the separated signals. They have shape (1, mixture length)
    """
    (x_0, x_1) = _ancestral_sample(
        priors=priors,
        mixture=mixture,
        n_samples=num_beams,
        likelihood=likelihood,
        temperature=temperature,
    )

    # Note: x_0 and x_1 have shape (n_samples, num_tokens)
    return x_0[-1:], x_1[-1:]


def _sample(posterior_data: torch.Tensor, coords: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    # check input shape
    batch_size, nnz_posterior = posterior_data.shape
    num_dims, nnz_coords = coords.shape

    assert num_dims == 2
    assert nnz_coords == nnz_posterior

    samples = torch.distributions.Categorical(logits=posterior_data).sample()
    x_0, x_1 = torch.gather(coords, dim=-1, index=samples.view(1, batch_size).repeat(num_dims, 1))
    return x_0, x_1


def fast_sampled_separation(
    priors: Sequence[SeparationPrior],
    likelihood: Likelihood,
    mixture: Sequence[int],
    num_samples: int,
    temperature: float,
    top_k: Optional[int] = None,
):
    """Separate the input mixture using sampling.

    Args:
        priors: List of priors to use for separation
        likelihood: Likelihood to use for separation
        mixture: Sequence of tokens to separate
        num_samples: number of beams to use for the beamsearch
        temperature: temperature of the priors
        top_k: top-k value to use during sampling

    Returns:
        (s0, s1) the separated signals. They have shape (1, mixture length)
    """
    prior_0, prior_1 = priors
    sample_tokens = len(mixture)

    # check that everything is on the same device
    assert len(set([p._get_device() for p in priors] + [likelihood.get_device()])) == 1
    device = likelihood.get_device()

    # initialize loop variables
    xs_0, xs_1 = torch.full((2, num_samples, sample_tokens + 1), fill_value=-1, dtype=torch.long, device=device)
    xs_0[:, 0], xs_1[:, 0] = [p.get_sos() for p in priors]

    # loop over all samples tokens
    past_0, past_1 = None, None
    for sample_t in range(sample_tokens):

        # compute log priors
        log_p_0, past_0 = prior_0._get_logits(xs_0[:, : sample_t + 1], past_0)
        log_p_1, past_1 = prior_1._get_logits(xs_1[:, : sample_t + 1], past_1)

        # NOTE: during token separation batch-size should be equal to num. samples
        assert len(log_p_0) == len(log_p_1) == num_samples
        assert log_p_0.shape[-1] == log_p_1.shape[-1] == likelihood.get_tokens_count()

        # normalize priors and apply temperature
        log_p_0 = normalize_logits(log_p_0, temperature)
        log_p_1 = normalize_logits(log_p_1, temperature)

        # log likelihood in sparse COO format
        assert isinstance(mixture[sample_t], int)
        ll_coords, ll_data = likelihood._get_log_likelihood(mixture[sample_t])

        # compute log posterior
        if ll_coords.numel() > 0:
            # Note: posterior_data has shape (n_samples, nonzeros)
            posterior_data = _compute_log_posterior(ll_data, ll_coords, log_p_0, log_p_1)

            # apply topk filtering
            if top_k is not None:
                topk_values, _ = torch.topk(posterior_data, k=top_k, dim=-1)
                indices_to_remove = posterior_data < topk_values[..., -1:]
                posterior_data = posterior_data.masked_fill(indices_to_remove, -np.inf)

            x_0, x_1 = _sample(posterior_data, ll_coords)
        else:
            raise RuntimeError(f"Code {mixture[sample_t]} is not available in likelihood!")

        # Note: x_0, x_1 have shape (n_sample,)
        xs_0[:, sample_t + 1] = x_0
        xs_1[:, sample_t + 1] = x_1

    result_0, result_1 = xs_0[:, 1:], xs_1[:, 1:]

    # check returned separation is correctly masked
    assert (result_0 == -1).sum() == 0
    assert (result_1 == -1).sum() == 0

    return result_0, result_1
