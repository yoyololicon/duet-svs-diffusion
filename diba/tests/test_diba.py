#!/usr/bin/env python

"""Tests for `diba` package."""
from itertools import product
from typing import Any

import pytest
import torch
from click.testing import CliRunner

from diba import cli
from diba.diba import beamsearch_separation
from diba.interfaces import Likelihood, SeparationPrior


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "diba.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


class MarkovPrior(SeparationPrior):
    def __init__(self, probs: torch.Tensor):
        self._probs = probs
        assert probs.shape[0] == probs.shape[1]

    def get_tokens_count(self):
        return self._probs.shape[0]

    def get_sos(self):
        return torch.zeros(1, dtype=torch.long)

    def get_logits(self, token_idx: torch.Tensor, cache: Any = None):
        return torch.log(self._probs[token_idx[:, -1], :] + 1e-16), None

    def reorder_cache(self, cache: Any, beam_idx: torch.LongTensor) -> Any:
        pass

    def clean_cache(self):
        pass

    def get_device(self):
        return "cpu"


class SumUniformLikelihood(Likelihood):
    def __init__(self, token_count: int):
        self.tokens_count = token_count
        super().__init__()

    def get_log_likelihood(self, token_idx: int):
        coords = [(i, j) for i, j in product(range(token_idx + 1), range(token_idx + 1)) if i + j == token_idx]
        data = torch.tensor([1.0 / len(coords)] * len(coords), dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.long).T
        return coords, torch.log(data)

    def get_device(self):
        return torch.device("cpu")

    def get_tokens_count(self):
        return self.tokens_count


def test_beamsearch():

    with torch.no_grad():
        x0 = torch.tensor([0, 1, 1], dtype=torch.long)
        x1 = torch.tensor([0, 0, 1], dtype=torch.long)

        mixture = [0, 1, 2]

        # Note: the mixture can be obtained only with the following signals:
        # [0,0,0] + [0,1,2]
        # [0,0,1] + [0,1,1]
        # [0,0,2] + [0,1,0]
        # [0,1,0] + [0,0,2]
        # [0,1,1] + [0,0,1]
        # [0,1,2] + [0,0,0]

        p0 = MarkovPrior(
            torch.tensor(
                [
                    [0.9, 0.1, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        # Note: This markov chain can be seen as
        # 0 -> 0 (with prob. 0.9)
        # 0 -> 1 (with prob. 0.1)
        # 1 -> 1 (always)
        # 2 -> 2 (always)

        p1 = MarkovPrior(
            torch.tensor(
                [
                    [0.1, 0.9, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        # Note: This markov chain can be seen as
        # 0 -> 0 (with prob. 0.1)
        # 0 -> 1 (with prob. 0.9)
        # 1 -> 0 (always)
        # 2 -> 2 (always)

        # Note: with the above separation should work only
        # when the number of beams is >= 2. Otherwise it falls
        # in a local maximum.

        for n_samples in [1, 2]:
            r0, r1 = beamsearch_separation(
                priors=[p0, p1], likelihood=SumUniformLikelihood(3), mixture=mixture, num_samples=n_samples
            )

            r0, r1 = torch.tensor(r0), torch.tensor(r1)
            assert torch.equal(r0 + r1, torch.tensor(mixture)), f"{(r0+r1).tolist()} != {x1.tolist()} (expected)"

            r0_eq, r1_eq = torch.equal(r0, x0), torch.equal(r1, x1)

            # it should be able to correctly separate only with n_samples > 1
            if n_samples > 1:
                assert r0_eq, f"{r0.tolist()} != {x0.tolist()} (expected)"
                assert r1_eq, f"{r1.tolist()} != {x1.tolist()} (expected)"
            else:
                assert not r0_eq, f"{r0.tolist()} == {x0.tolist()} (expected different result)"
                assert not r1_eq, f"{r1.tolist()} == {x1.tolist()} (expected different result)"
