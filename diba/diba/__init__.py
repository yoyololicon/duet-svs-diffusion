"""Top-level package for Discrete Bayesian Signal Separation."""
from .diba import fast_beamsearch_separation, fast_sampled_separation, separate
from .interfaces import Likelihood, SeparationPrior

__author__ = """Giorgio Mariani"""
__email__ = "giorgiomariani94@gmail.com"
__version__ = "0.1.0"

__all__ = (
    fast_beamsearch_separation,
    fast_sampled_separation,
    Likelihood,
    SeparationPrior,
    separate,
)
