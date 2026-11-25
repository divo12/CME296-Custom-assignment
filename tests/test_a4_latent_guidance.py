from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from tests.adapters import (
    run_cfg_combine,
    run_classifier_guidance_score,
    run_clip_loss,
    run_latent_scaler,
    run_siglip_loss,
    run_vae_kl_loss,
    run_vae_reparameterize,
)


def test_vae_kl_and_reparameterization():
    # Standard normal posterior N(0, I) -> KL == 0
    zeros = torch.zeros((4, 4, 8, 8))
    assert torch.allclose(run_vae_kl_loss(zeros, zeros), torch.tensor(0.0), atol=1e-7)

    # Analytical check on 1D vector latents: mu = 2, var = e^1 -> 0.5 * (4 + e - 1 - 1) = 1 + e/2 per dim
    mu = torch.full((3, 5), 2.0)
    logvar = torch.ones((3, 5))
    expected_kl = 5.0 * 0.5 * (4.0 + math.e - 2.0)
    assert torch.allclose(run_vae_kl_loss(mu, logvar), torch.tensor(expected_kl), atol=1e-5)

    eps = torch.ones_like(mu) * 0.5
    z = run_vae_reparameterize(mu, logvar, eps=eps, deterministic=False)
    assert torch.allclose(z, mu + math.exp(0.5) * eps, atol=1e-6)
    z_det = run_vae_reparameterize(mu, logvar, eps=eps, deterministic=True)
    assert torch.allclose(z_det, mu, atol=1e-7)


def test_latent_scaler_unit_variance():
    g = torch.Generator().manual_seed(55)
    unscaled = torch.randn((32, 4, 8, 8), generator=g) * 5.4899  # SD-like VAE spread
    scaled, factor = run_latent_scaler(unscaled)
    assert abs(factor - (1.0 / 5.4899)) < 0.02
    assert abs(scaled.std().item() - 1.0) < 1e-5


def test_clip_loss_accuracy_clamping_and_stability():
    g = torch.Generator().manual_seed(101)
    img = torch.randn((6, 16), generator=g) * 3.0
    txt = torch.randn((6, 16), generator=g) * 0.5
    logit_scale_log = torch.tensor(math.log(14.0), requires_grad=True)

    loss = run_clip_loss(img, txt, logit_scale_log)
    img_n = F.normalize(img, dim=-1)
    txt_n = F.normalize(txt, dim=-1)
    logits_ref = 14.0 * (img_n @ txt_n.T)
    labels = torch.arange(6)
    ref = 0.5 * (F.cross_entropy(logits_ref, labels) + F.cross_entropy(logits_ref.T, labels))
    assert torch.allclose(loss, ref, atol=1e-5)
    loss.backward()
    assert logit_scale_log.grad is not None and torch.isfinite(logit_scale_log.grad)

    # Test temperature clamp at 100.0 (exp(10.0) -> clamped to 100.0, no overflow/NaN!)
    huge_scale = torch.tensor(10.0)
    loss_clamped = run_clip_loss(img, img.clone(), huge_scale)
    assert torch.isfinite(loss_clamped)


def test_siglip_loss_pairwise_independence_and_stability():
    g = torch.Generator().manual_seed(202)
    img = torch.randn((5, 12), generator=g)
    txt = torch.randn((5, 12), generator=g)
    t_log = torch.tensor(math.log(10.0), requires_grad=True)
    bias = torch.tensor(-10.0, requires_grad=True)

    loss = run_siglip_loss(img, txt, t_log, bias)
    img_n = F.normalize(img, dim=-1)
    txt_n = F.normalize(txt, dim=-1)
    logits = 10.0 * (img_n @ txt_n.T) - 10.0
    z = 2.0 * torch.eye(5) - 1.0
    ref = -F.logsigmoid(z * logits).sum() / 5.0
    assert torch.allclose(loss, ref, atol=1e-5)
    loss.backward()
    assert t_log.grad is not None and bias.grad is not None


def test_classifier_guidance_and_cfg_rescaling():
    W = torch.tensor([[2.0, -1.0], [-1.0, 3.0], [0.5, 0.5]])  # 3 classes in R^2

    def clf_log_probs(x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x @ W.T, dim=-1)

    xt = torch.tensor([[1.0, -0.5], [0.2, 1.5]])
    labels = torch.tensor([0, 1])
    uncond = torch.zeros_like(xt)

    guided = run_classifier_guidance_score(uncond, xt, labels, clf_log_probs, guidance_scale=2.5)
    probs = F.softmax(xt @ W.T, dim=-1)
    # Exact analytical grad_x log softmax(x W^T)_c = W_c - sum_k p_k W_k
    expected_grad = W[labels] - probs @ W
    assert torch.allclose(guided, 2.5 * expected_grad, atol=1e-5)

    # Test CFG + Lin et al. Std-Rescaling prevents variance blowup at high w = 12.0!
    g = torch.Generator().manual_seed(77)
    cond_pred = torch.randn((4, 3, 8, 8), generator=g)
    uncond_pred = torch.randn((4, 3, 8, 8), generator=g)

    cfg_raw = run_cfg_combine(cond_pred, uncond_pred, cfg_scale=12.0, rescale_phi=0.0)
    cfg_rescaled = run_cfg_combine(cond_pred, uncond_pred, cfg_scale=12.0, rescale_phi=1.0)

    assert torch.allclose(cfg_raw, uncond_pred + 12.0 * (cond_pred - uncond_pred), atol=1e-5)
    # At rescale_phi = 1.0, per-sample std of cfg_rescaled must match cond_pred's per-sample std!
    assert torch.allclose(
        cfg_rescaled.std(dim=(1, 2, 3)),
        cond_pred.std(dim=(1, 2, 3)),
        atol=1e-4,
    )
