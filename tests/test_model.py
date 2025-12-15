"""Test model.py"""

import pytest

import torch
from sisal.model import BetaVAE, BetaVAESynthetic, View, l_out

def test_betavae_init():
    model = BetaVAE(2, 100)
    assert model, "Expected instantiated model"
    assert model.z_dim == 2, "z_dim should be 2"
    assert model.encoder is not None, "Encoder was None"
    assert model.decoder is not None, "Decoder was None"

    with pytest.raises(ValueError):
        BetaVAE(2, 10)


def test_syntheticbetavae_init():
    model = BetaVAESynthetic(2, 100)
    assert model, "Expected instantiated model"
    assert model.z_dim == 2, "z_dim should be 2"
    assert model.encoder is not None, "Encoder was None"
    assert model.decoder is not None, "Decoder was None"
    

def test_l_out():
    # Test cases manually calculated or known
    # l_out(l_in, ker, stride, pad) = floor((l_in + 2*pad - ker) / stride + 1)
    assert l_out(100, 10, 2, 1) == 47
    assert l_out(46, 10, 2, 1) == 20


def test_view_layer():
    v = View((-1, 10))
    t = torch.randn(5, 2, 5)  # 5 * 10 elements
    out = v(t)
    assert out.shape == (5, 10)


def test_betavae_forward():
    model = BetaVAE(z_dim=5, in_size=100)
    # Batch size 3, 1 channel, signal length 100
    x = torch.randn(3, 1, 100)
    mu, logvar = model(x)
    assert mu.shape == (3, 5)
    assert logvar.shape == (3, 5)


def test_betavae_synthetic_forward():
    model = BetaVAESynthetic(z_dim=5, in_size=20) 
    x = torch.randn(3, 1, 5)
    mu, logvar = model(x)
    assert mu.shape == (3, 5)
    assert logvar.shape == (3, 5)
