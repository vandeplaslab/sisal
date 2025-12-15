"""Test utils.py"""
import pytest
import torch
import numpy as np
from sisal.utils import reparametrize, normalize_train_test_full_loader, compute_latent_mean

def test_reparametrize():
    mu = torch.zeros(10, 2)
    logvar = torch.zeros(10, 2)
    
    # Test shape
    z = reparametrize(mu, logvar)
    assert z.shape == (10, 2)
    
    # Test stochasticity (sanity check, variance should be non-zero mostly)
    # If we set seed, we should get same result
    torch.manual_seed(42)
    z1 = reparametrize(mu, logvar)
    torch.manual_seed(42)
    z2 = reparametrize(mu, logvar)
    assert torch.allclose(z1, z2)
    
    # Different seeds
    torch.manual_seed(43)
    z3 = reparametrize(mu, logvar)
    assert not torch.allclose(z1, z3)

def test_normalize_train_test_full_loader():
    # Create dummy data
    n_points = 100
    n_dim = 10
    centroids = np.random.randn(n_points, n_dim)
    mask = np.ones(n_points)
    
    train_loader, test_loader, full_loader = normalize_train_test_full_loader(
        centroids, mask, batch_size=10
    )
    
    # Default split 0.8 / 0.2
    # Train size should be around 80
    # Test size around 20
    # Note: drop_last=True for train/test loaders in utils.py
    
    assert len(train_loader.dataset) == 80
    assert len(test_loader.dataset) == 20
    assert len(full_loader.dataset) == 100
    
    # Check batch shapes
    # Train loader yields [x, mask]
    x, m = next(iter(train_loader))
    # Shape of x: (batch_size, 1, n_dim) in code: v = np.array(v, dtype="float32")[:, np.newaxis, :]
    assert x.shape == (10, 1, n_dim)
    assert m.shape == (10,)

def test_compute_latent_mean():
    from unittest.mock import MagicMock
    
    model = MagicMock()
    # Mock forward return value: (z_mean, z_logvar)
    # We call it with a loader.
    # Loader yields (x, _)
    x = torch.randn(2, 1, 100)
    # Loader needs to be iterable AND have a .dataset property
    loader = MagicMock()
    loader.__iter__.return_value = iter([(x, None)] * 2)
    loader.dataset = [0] * 4 # length 4
    
    model.z_dim = 5
    # When forward is called, return z_mean with shape (batch, z_dim)
    model.forward.return_value = (torch.zeros(2, 5), torch.zeros(2, 5))
    
    latent_means = compute_latent_mean(loader, model)
    
    assert latent_means.shape == (4, 5) # 2 batches * 2 samples
    assert np.all(latent_means == 0)
