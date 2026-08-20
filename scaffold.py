"""
Mamba from Scratch: Selective State Spaces scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Tiny character-level Mamba LM: train a few steps, then greedily generate."""
import numpy as np
import torch


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    corpus = "abacabadabacaba abacabadabacaba"
    prompt = "aba"
    generated = train_tiny_mamba_and_generate(
        corpus,
        n_steps=8,
        lr=0.05,
        prompt=prompt,
        max_new_tokens=8,
        d_model=16,
        n_layers=2,
        d_state=4,
        d_inner=32,
        conv_kernel=3,
        seed=0,
    )
    print("corpus_len", len(corpus))
    print("prompt", prompt)
    print("generated", generated)


if __name__ == "__main__":
    main()
