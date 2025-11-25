"""Test model.py"""

import pytest

from sisal.model import BetaVAE, BetaVAESynthetic


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

    with pytest.raises(ValueError):
        BetaVAESynthetic(2, 10)
