"""Test solver."""

from sisal.solver import Solver


def test_solver_init():
    solver = Solver(beta=0.1, z_dim=2, in_size=100, epochs=1, device="cpu")
    assert solver, "Expected instantiated solver"
    assert solver.beta == 0.1, "Beta should be 0.1"
    assert solver.z_dim == 2, "z_dim should be 2"
    assert solver.epochs == 1, "Epochs should be 1"
    assert solver.device.type == "cpu", "Device should be cpu"


def test_kl_divergence():
    solver = Solver(beta=1.0, z_dim=2, in_size=100, epochs=1, device="cpu")
    import torch
    
    # KL = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # if mu=0, logvar=0 (var=1), KL should be 0
    mu = torch.zeros(1, 2)
    logvar = torch.zeros(1, 2)
    kl = solver.KL(mu, logvar)
    assert torch.isclose(kl, torch.tensor(0.0)), f"Expected KL 0, got {kl}"

    # if mu=1, logvar=0, KL = 0.5 * (1 + 0 - 1 - 1) = -0.5 ?? No formula is 0.5 * (mu^2 + exp(logvar) - logvar - 1)
    # Check code: klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # WAIT, let's check the code implementation in Solver.KL
    # Code says: klds = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
    # which matches standard KL(N(mu, var) || N(0, 1))
    
    mu = torch.ones(1, 2)
    kl = solver.KL(mu, logvar)
    # 0.5 * (1 + 1 - 0 - 1) = 0.5 per dimension. Sum(1) -> 1.0. Mean(0) -> 1.0
    assert torch.isclose(kl, torch.tensor(1.0)), f"Expected KL 1.0, got {kl}"


def test_reconstruction_loss():
    solver = Solver(beta=1.0, z_dim=2, in_size=100, epochs=1, device="cpu")
    import torch
    
    x = torch.zeros(10, 1, 100)
    recon = torch.zeros(10, 1, 100)
    # loss = 0.5 * MSE * (sum / batch_size) ? 
    # Code: r_loss = 0.5 * F.mse_loss(x, mu_x, reduction="sum").div(batch_size)
    # MSE sum of zeros is 0
    loss = solver.reconstruction_loss(x, recon)
    assert loss == 0

    x = torch.ones(1, 1, 100)
    recon = torch.zeros(1, 1, 100)
    # MSE sum = 100. div(1) = 100. * 0.5 = 50
    loss = solver.reconstruction_loss(x, recon)
    assert loss == 50.0


def test_loss_function():
    solver = Solver(beta=2.0, z_dim=2, in_size=100, epochs=1, device="cpu")
    import torch
    
    x = torch.zeros(1, 1, 100)
    z_mean = torch.zeros(1, 2)
    z_logvar = torch.zeros(1, 2)
    decoder_mean = torch.zeros(1, 1, 100)
    
    # recon = 0, KL = 0 -> total = 0
    loss = solver.loss(beta=2.0, x=x, z_mean=z_mean, z_logvar=z_logvar, decoder_mean=decoder_mean)
    assert loss == 0


def test_train_one_epoch_mock():
    from unittest.mock import MagicMock
    import torch
    
    solver = Solver(beta=1.0, z_dim=2, in_size=100, epochs=1, device="cpu")
    
    # Mock dataloader
    # Yields (x, y) tuples
    x = torch.randn(2, 1, 100)
    y = torch.tensor([0, 1])
    dataloader = [(x, y)] * 2 # 2 batches
    
    solver.model = MagicMock(wraps=solver.model)
    
    loss, recons, kl = solver.train_one_epoch(0, dataloader)
    
    assert solver.model.forward.called
    # It should run without error and return metrics
    # Since loss is accumulated and averaged every 1000 batches, and we only have 2, 
    # last_loss will be 0.0 because the print/update condition i % 1000 == 999 is not met.
    # Logic check:
    # if i % 1000 == 999: last_loss = running_loss/1000 ...
    # return last_loss
    # So for small dataloader, it returns 0.
    
    assert loss == 0.0
