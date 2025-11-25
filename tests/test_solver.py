"""Test solver."""

from sisal.solver import Solver


def test_solver_init():
    solver = Solver(beta=0.1, z_dim=2, in_size=100, epochs=1, device="cpu")
    assert solver, "Expected instantiated solver"
    assert solver.beta == 0.1, "Beta should be 0.1"
    assert solver.z_dim == 2, "z_dim should be 2"
    assert solver.epochs == 1, "Epochs should be 1"
    assert solver.device.type == "cpu", "Device should be cpu"
