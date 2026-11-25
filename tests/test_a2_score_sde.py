from __future__ import annotations

import math
import torch
from torch import nn

from tests.adapters import (
    run_annealed_langevin_dynamics,
    run_dsm_loss,
    run_ism_hutchinson_loss,
    run_pf_ode_drift,
    run_reverse_sde_step,
    run_ve_sde_params,
    run_vp_sde_params,
)


def test_ism_hutchinson_exact_linear_score_and_grad():
    """
    For s_theta(x) = -A x where A = diag(a_1, ..., a_d),
    tr(grad_x s_theta(x)) = -sum(a_j) for ANY Rademacher probe vector v in {-1, +1}^d!
    """
    d = 6
    log_diag = nn.Parameter(torch.linspace(-0.5, 0.8, d))

    def score_fn(x: torch.Tensor) -> torch.Tensor:
        return -x * torch.exp(log_diag)

    g = torch.Generator().manual_seed(42)
    x = torch.randn((12, d), generator=g)
    # Rademacher probes give ZERO variance on a diagonal Jacobian!
    probes = torch.randint(0, 2, (12, d), generator=g).float() * 2.0 - 1.0

    loss = run_ism_hutchinson_loss(score_fn, x, probes)
    a = torch.exp(log_diag)
    expected = (-a.sum() + 0.5 * ((x * a) ** 2).sum(dim=-1)).mean()
    assert torch.allclose(loss, expected, atol=1e-5)

    # Must support second-order backprop into model parameters!
    loss.backward()
    assert log_diag.grad is not None and torch.isfinite(log_diag.grad).all()
    assert log_diag.grad.abs().sum() > 0


def test_dsm_loss_variance_balanced():
    g = torch.Generator().manual_seed(11)
    x_clean = torch.randn((8, 3, 4, 4), generator=g)
    sigma = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    eps = torch.randn_like(x_clean)
    x_noisy = x_clean + sigma.view(8, 1, 1, 1) * eps

    # Exact oracle score gives strictly zero DSM loss across 500x range of sigmas
    oracle_score = -eps / sigma.view(8, 1, 1, 1)
    zero_loss = run_dsm_loss(oracle_score, x_clean, x_noisy, sigma)
    assert torch.allclose(zero_loss, torch.tensor(0.0), atol=1e-6)

    # Zero score prediction gives E[sigma^2 * ||eps / sigma||^2] = E[eps^2] ~ 1.0 (not 1/sigma^2!)
    unit_loss = run_dsm_loss(torch.zeros_like(oracle_score), x_clean, x_noisy, sigma)
    assert torch.allclose(unit_loss, (eps ** 2).mean(), atol=1e-6)


def test_annealed_langevin_dynamics_converges_to_gaussian_mean():
    """With zero injected noise, ALD on N(mu, I) score moves monotonically toward mu."""
    target_mu = torch.tensor([[3.0, -2.0], [-1.5, 4.0]])

    def exact_gaussian_score(x: torch.Tensor, sigma_batch: torch.Tensor) -> torch.Tensor:
        var = 1.0 + sigma_batch[:, None] ** 2
        return -(x - target_mu) / var

    x_init = torch.zeros((2, 2))
    sigmas = torch.tensor([2.0, 1.0, 0.5])
    zero_noise = torch.zeros((3, 15, 2, 2))

    x_final = run_annealed_langevin_dynamics(
        exact_gaussian_score,
        x_init=x_init,
        sigmas=sigmas,
        n_steps_per_sigma=15,
        step_lr=0.25,
        noise_tensors=zero_noise,
    )
    init_dist = (x_init - target_mu).norm(dim=-1)
    final_dist = (x_final - target_mu).norm(dim=-1)
    assert torch.all(final_dist < 0.05 * init_dist)


def test_vp_and_ve_sde_closed_form_kernels():
    x = torch.ones((4, 2, 3))
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])

    vp = run_vp_sde_params(x, t, beta_min=0.1, beta_max=20.0)
    # Boundary conditions at t = 0: alpha(0) = 1, std(0) = 0, g(0) = sqrt(0.1)
    assert torch.allclose(vp["mean_coeff"][0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(vp["std"][0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(vp["diffusion"][0], torch.tensor(math.sqrt(0.1)), atol=1e-6)
    # VP identity: alpha(t)^2 + std(t)^2 == 1 everywhere!
    assert torch.allclose(vp["mean_coeff"] ** 2 + vp["std"] ** 2, torch.ones_like(t), atol=1e-5)

    ve = run_ve_sde_params(x, t, sigma_min=0.01, sigma_max=50.0)
    assert torch.allclose(ve["drift"], torch.zeros_like(x), atol=1e-7)
    assert torch.allclose(ve["mean_coeff"], torch.ones_like(t), atol=1e-7)
    assert torch.allclose(ve["std"][0], torch.tensor(0.01), atol=1e-6)
    assert torch.allclose(ve["std"][-1], torch.tensor(50.0), atol=1e-4)


def test_reverse_sde_vs_pf_ode_factor_of_half():
    """
    Reverse SDE drift removes g(t)^2 * score, whereas PF-ODE removes 0.5 * g(t)^2 * score!
    """
    x = torch.randn((5, 3, 4))
    t = torch.full((5,), 0.6)
    drift = -2.0 * x
    diffusion = torch.full((5,), 3.0)
    score = torch.ones_like(x) * 0.5
    dt = 0.01

    pf_drift = run_pf_ode_drift(drift, diffusion, score)
    expected_pf = drift - 0.5 * (diffusion[:, None, None] ** 2) * score
    assert torch.allclose(pf_drift, expected_pf, atol=1e-6)

    # Zero-noise Euler-Maruyama reverse SDE step moves by -[f - g^2 s] * dt
    rev_det = run_reverse_sde_step(x, t, dt, drift, diffusion, score, z=torch.zeros_like(x))
    expected_rev = x - (drift - (diffusion[:, None, None] ** 2) * score) * dt
    assert torch.allclose(rev_det, expected_rev, atol=1e-6)
