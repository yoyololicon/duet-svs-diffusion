from typing import Tuple

import torch


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> Tuple[torch.LongTensor, ...]:
    """Convert flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).

    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    return tuple(coord[::-1])


def get_topk(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, Tuple[torch.LongTensor, ...]]:
    """Get top k elements in tensor.

    Args:
        x: A tensor of values of any shape

    Returns:
        A tuple containing:
        - Tensor containing the k greatest elements in ascending order. Shape: (k,)
        - Tuple of indices for the top k elements. The number of elements in the
         tuple is the same as the number of dimensions in token_idx

    """
    # log_post_sum, log_post_index = torch.topk(log_posterior.flatten(), n_samples)
    x_sorted, log_post_index = torch.sort(x.flatten(), dim=-1)
    x_sorted, log_post_index = x_sorted[-k:], log_post_index[-k:]
    idx = unravel_indices(log_post_index, x.shape)
    return x_sorted, idx


def normalize_logits(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Normalize Logits between (-inf, 0.0] with additional temperature.

    Args:
        x: Tensor containing the logits to normalize. This must have values
         between (-inf, inf) and shape (*, N)
        temperature: temperature to use in normalization

    Returns:
        The normalized logits between (-inf, 0.0], with the same shape as token_idx

    """
    return torch.distributions.Categorical(logits=x / temperature).logits
