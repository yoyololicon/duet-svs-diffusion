from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple, Union

import torch


class WrongShapeError(RuntimeError):
    def __init__(self, x: torch.Tensor, target_shape: Sequence[int]):
        target_shape_str = ",".join(str(d) if d != -1 else "*" for d in target_shape)
        x_shape_str = ",".join(str(d) for d in x.shape)
        super().__init__(f"Incorrect shape: expected ({target_shape_str}), got ({x_shape_str}) instead!")


def _check_shape(x: torch.Tensor, target_shape: Sequence[int]):
    if len(x.shape) != len(target_shape):
        raise WrongShapeError(x, target_shape=target_shape)

    for xdim, tdim in zip(x.shape, target_shape):
        if tdim != -1 and xdim != tdim:
            raise WrongShapeError(x, target_shape=target_shape)


def _check_dtype(x: torch.Tensor, target_dtype: Union[torch.dtype, Sequence[torch.dtype]]):
    if not isinstance(target_dtype, Sequence):
        target_dtype = [target_dtype]

    return any(x.dtype == dtype for dtype in target_dtype)


class DevicePortable:
    def _get_device(self) -> torch.device:
        return torch.device(self.get_device())

    @abstractmethod
    def get_device(self) -> torch.device:
        """Return the device currently used by the object."""
        ...


class Likelihood(ABC, DevicePortable):
    def _get_log_likelihood(self, x: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        loglikelihood = self.get_log_likelihood(x)
        if not isinstance(loglikelihood, tuple) and len(loglikelihood) == 2:
            raise RuntimeError(f"Incorrect value: expected tuple of two tensors, got {type(loglikelihood)} instead!")

        coords, data = loglikelihood
        _check_dtype(coords, torch.long)
        _check_dtype(data, [torch.float16, torch.float32])

        _check_shape(coords, [2, -1])
        _check_shape(data, [-1])

        if data.shape[0] != coords.shape[-1]:
            raise RuntimeError(
                "Incompatible shapes: coords and data have a different "
                f"number of elements: {coords.shape[-1]} != { data.shape[0]}"
            )

        if data.numel() == 0:
            return coords, data

        if data.max() > 0.0:
            raise RuntimeError(f"Unexpected value: loglikelihood value is greater than zero: {data.max()} > 0.0")

        return coords, data

    @abstractmethod
    def get_log_likelihood(self, token_idx: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Return the loglikelihood for the input mixture token.

        Args:
            token_idx: token of the mixture from which obtain the likelihood

        Returns:
            A tuple containing:
            - The coordinates of nonzero elements in the likelihood. Shape: (2, num nonzero)
            - The loglikelihood values for the above coordinates. Shape: (num nonzero,)

        """
        ...

    @abstractmethod
    def get_tokens_count(self) -> int:
        """Return the number of tokens used by the log-likelihood."""
        ...


class SeparationPrior(ABC, DevicePortable):
    def _reorder_cache(self, cache: Any, beam_idx: torch.LongTensor) -> Any:
        return self.reorder_cache(cache, beam_idx)

    def _get_logits(
        self,
        x: torch.LongTensor,
        past_key_value: Any = None,
    ) -> torch.Tensor:
        logits, past_kv = self.get_logits(x, past_key_value)

        _check_shape(logits, [-1, self.get_tokens_count()])
        _check_dtype(logits, [torch.float16, torch.float32])

        return logits, past_kv

    @abstractmethod
    def get_tokens_count(self) -> int:
        """Return the number of tokens used by the log-likelihood."""
        ...

    @abstractmethod
    def get_sos(self) -> Any:
        """Return the start of sentence token.

        This token is then passed as argument for the first call of 'get_logits'.
        """
        ...

    @abstractmethod
    def get_logits(
        self,
        token_ids: torch.LongTensor,
        cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Return the logits for the input token.

        Compute the logits for the next token probabilities, given the input token.

        Args:
            token_ids: token prefix from which compute the next token logits. It
             should have shape (batch size, sequence length).

            cache: past keys and values used by the transformer

        Returns:
            * Logits for the next tokens, should have shape (batch size, num tokens)
            * Cache to use for the next token
        """
        ...

    def reorder_cache(self, cache: Any, beam_idx: torch.LongTensor) -> Any:
        """Reorder the cache values using the input beams.

        It is necessary to reimplement this method when using beam search with cached values.

        Args:
            cache: past values and keys of the transformer
            beam_idx: beams indices, has shape: (num. beams,)

        Returns:
            reordered cache w.r.t. the beam indices.
        """
        raise NotImplementedError("Implement method in order to use beam-search with cached values!")
