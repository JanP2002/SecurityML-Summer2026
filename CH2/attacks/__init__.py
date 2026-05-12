"""
attacks/__init__.py — Public API for the contamination methods library.

Import individual transform functions (return tensors) or dataset
factory functions (return TensorDataset with label=1) from here.

Example
-------
    from attacks import make_gaussian_attack, make_backdoor_attack
    from attacks.contamination import geometric_distortion  # raw tensor API
"""

from .contamination import (
    # ---- Raw tensor transforms (input: Tensor, output: Tensor) ----
    gaussian_noise,
    salt_and_pepper,
    geometric_distortion,
    blended_attack,
    backdoor_trigger,
    # ---- TensorDataset factories (label=1 applied automatically) ----
    make_gaussian_attack,
    make_salt_pepper_attack,
    make_geometric_attack,
    make_blended_attack,
    make_backdoor_attack,
    make_ood_attack,
)

__all__ = [
    "gaussian_noise",
    "salt_and_pepper",
    "geometric_distortion",
    "blended_attack",
    "backdoor_trigger",
    "make_gaussian_attack",
    "make_salt_pepper_attack",
    "make_geometric_attack",
    "make_blended_attack",
    "make_backdoor_attack",
    "make_ood_attack",
]
