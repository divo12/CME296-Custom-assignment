from __future__ import annotations

import math
import torch

from tests.adapters import (
    run_convert_predictions,
    run_ddim_step,
    run_ddpm_forward,
    run_ddpm_loss,
    run_ddpm_posterior,
    run_ddpm_step,
    run_noise_schedule,
)


def test_linear_and_cosine_schedules():
    T = 1000
    lin = run_noise_schedule("linear", T=T, beta_start=1e-4, beta_end=0.02)
    for k in ("betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev", "log_snr"):
        assert k in lin and lin[k].shape == (T,)

    assert torch.allclose(lin["betas"][0], torch.tensor(1e-4), atol=1e-7)
    assert torch.allclose(lin["betas"][-1], torch.tensor(0.02), atol=1e-7)
    assert torch.allclose(lin["alphas"], 1.0 - lin["betas"], atol=1e-7)
    assert torch.allclose(lin["alphas_cumprod"], torch.cumprod(lin["alphas"], dim=0), atol=1e-6)
    assert torch.allclose(lin["alphas_cumprod_prev"][0], torch.tensor(1.0), atol=1e-7)
    assert torch.allclose(lin["alphas_cumprod_prev"][1:], lin["alphas_cumprod"][:-1], atol=1e-7)
    # log-SNR must strictly decrease across time
    assert torch.all(lin["log_snr"][1:] < lin["log_snr"][:-1])

    cos = run_noise_schedule("cosine", T=T, cosine_s=0.008)
    assert torch.all((cos["betas"] > 0.0) & (cos["betas"] <= 0.999 + 1e-7))
    assert torch.allclose(cos["alphas"], 1.0 - cos["betas"], atol=1e-6)
    assert torch.allclose(cos["alphas_cumprod"], torch.cumprod(cos["alphas"], dim=0), atol=1e-6)
    # Cosine preserves substantially more signal at t = T // 2 than linear
    assert cos["alphas_cumprod"][T // 2] > lin["alphas_cumprod"][T // 2] + 0.15


def test_ddpm_forward_shapes_and_vp_variance():
    sched = run_noise_schedule("cosine", T=200)
    g = torch.Generator().manual_seed(123)
    x0 = torch.randn((16, 3, 8, 8), generator=g)
    noise = torch.randn(x0.shape, generator=g)
    t_idx = torch.randint(0, 200, (16,), generator=g)

    xt = run_ddpm_forward(x0, t_idx, sched["alphas_cumprod"], noise)
    assert xt.shape == x0.shape

    ab = sched["alphas_cumprod"][t_idx].view(16, 1, 1, 1)
    expected = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
    assert torch.allclose(xt, expected, atol=1e-6)


def test_ddpm_posterior_bayes_consistency():
    """Verify q(x_{t-1} | x_t, x_0) matches analytical Bayes rule & step-0 logvar clipping."""
    sched = run_noise_schedule("linear", T=50, beta_start=1e-3, beta_end=0.05)
    g = torch.Generator().manual_seed(7)
    x0 = torch.randn((6, 2, 4, 4), generator=g)
    xt = torch.randn((6, 2, 4, 4), generator=g)
    t_idx = torch.tensor([0, 1, 5, 19, 35, 49], dtype=torch.long)

    post_mean, post_var, post_log_var = run_ddpm_posterior(
        x0, xt, t_idx, sched["betas"], sched["alphas"], sched["alphas_cumprod"]
    )
    assert post_mean.shape == x0.shape
    assert post_var.shape == (6, 1, 1, 1)
    assert post_log_var.shape == (6, 1, 1, 1)

    # At t_idx == 0, alpha_bar_prev = 1 -> posterior_var == 0 and posterior_mean == x0
    assert torch.allclose(post_var[0], torch.zeros_like(post_var[0]), atol=1e-7)
    assert torch.allclose(post_mean[0], x0[0], atol=1e-5)
    # Clipped log-variance at t_idx == 0 must equal log-variance at t_idx == 1 (finite!)
    assert torch.isfinite(post_log_var).all()
    assert torch.allclose(post_log_var[0], post_log_var[1], atol=1e-6)


def test_prediction_conversions_roundtrip():
    sched = run_noise_schedule("cosine", T=100)
    g = torch.Generator().manual_seed(99)
    x0 = torch.randn((8, 4, 6), generator=g)
    eps = torch.randn((8, 4, 6), generator=g)
    t_idx = torch.tensor([1, 10, 25, 40, 55, 70, 85, 98], dtype=torch.long)

    xt = run_ddpm_forward(x0, t_idx, sched["alphas_cumprod"], eps)
    ab = sched["alphas_cumprod"][t_idx].view(8, 1, 1)
    v_true = torch.sqrt(ab) * eps - torch.sqrt(1.0 - ab) * x0

    for pred_type, tensor in [("eps", eps), ("x0", x0), ("v", v_true)]:
        out = run_convert_predictions(xt, t_idx, sched["alphas_cumprod"], tensor, pred_type)
        assert torch.allclose(out["eps"], eps, atol=1e-4)
        assert torch.allclose(out["x0"], x0, atol=1e-4)
        assert torch.allclose(out["v"], v_true, atol=1e-4)


def test_min_snr_equivalence_across_parameterizations():
    """
    Key theoretical identity: under Min-SNR-gamma weighting, predicting eps, x0, or v
    from the SAME implied x0_hat produces the EXACT same weighted loss!
    """
    sched = run_noise_schedule("cosine", T=100)
    g = torch.Generator().manual_seed(2026)
    x0 = torch.randn((10, 3, 4, 4), generator=g)
    eps = torch.randn_like(x0)
    t_idx = torch.randint(1, 95, (10,), generator=g)
    xt = run_ddpm_forward(x0, t_idx, sched["alphas_cumprod"], eps)

    # Perturb x0 slightly to simulate an imperfect neural net prediction x0_hat
    x0_hat = x0 + 0.25 * torch.randn_like(x0)
    implied = run_convert_predictions(xt, t_idx, sched["alphas_cumprod"], x0_hat, "x0")

    loss_x0 = run_ddpm_loss(
        implied["x0"], x0, eps, t_idx, sched["alphas_cumprod"], pred_type="x0", weight_type="min_snr", min_snr_gamma=5.0
    )
    loss_eps = run_ddpm_loss(
        implied["eps"], x0, eps, t_idx, sched["alphas_cumprod"], pred_type="eps", weight_type="min_snr", min_snr_gamma=5.0
    )
    loss_v = run_ddpm_loss(
        implied["v"], x0, eps, t_idx, sched["alphas_cumprod"], pred_type="v", weight_type="min_snr", min_snr_gamma=5.0
    )

    assert torch.allclose(loss_x0, loss_eps, atol=1e-4, rtol=1e-4)
    assert torch.allclose(loss_x0, loss_v, atol=1e-4, rtol=1e-4)


def test_ddpm_and_ddim_step_equivalence_and_zero_step():
    """When t_prev = t - 1 and eta = 1.0, DDIM step must EXACTLY match ancestral DDPM step!"""
    sched = run_noise_schedule("linear", T=50)
    g = torch.Generator().manual_seed(314)
    xt = torch.randn((4, 2, 4, 4), generator=g)
    eps_pred = torch.randn_like(xt)
    z = torch.randn_like(xt)

    t_idx = torch.tensor([10, 20, 30, 49], dtype=torch.long)
    t_prev = t_idx - 1

    step_ddpm = run_ddpm_step(
        xt, t_idx, eps_pred, sched["betas"], sched["alphas"], sched["alphas_cumprod"], z
    )
    step_ddim_eta1 = run_ddim_step(
        xt, t_idx, t_prev, eps_pred, sched["alphas_cumprod"], eta=1.0, z=z
    )
    assert torch.allclose(step_ddpm, step_ddim_eta1, atol=1e-5)

    # At t_idx = 0 (t_prev = -1), both DDPM and deterministic DDIM must ignore z and recover x0_hat
    t0 = torch.zeros(4, dtype=torch.long)
    tm1 = torch.full((4,), -1, dtype=torch.long)
    out_ddpm_0 = run_ddpm_step(
        xt, t0, eps_pred, sched["betas"], sched["alphas"], sched["alphas_cumprod"], torch.randn_like(xt) * 100.0
    )
    out_ddim_0 = run_ddim_step(
        xt, t0, tm1, eps_pred, sched["alphas_cumprod"], eta=0.0, z=None
    )
    assert torch.allclose(out_ddpm_0, out_ddim_0, atol=5e-4)
