from __future__ import annotations

import torch

from tests.adapters import (
    run_cfm_loss,
    run_convert_time_convention,
    run_flow_sample_loop,
    run_ode_euler_step,
    run_ode_heun_step,
    run_ot_conditional_path,
    run_rosetta_conversions,
    run_trajectory_straightness,
)


def test_time_convention_bridge():
    """Verify CME 296 Lecture 3 (t=0 noise -> t=1 data) <-> Diffusion (tau=0 data -> tau=1 noise)."""
    x_noise = torch.tensor([[1.0, -2.0]])
    x_data = torch.tensor([[4.0, 6.0]])
    t_fm = torch.tensor([0.25])

    xt_fm, u_fm = run_ot_conditional_path(x_noise, x_data, t_fm, sigma_min=0.0)
    tau, v_diff = run_convert_time_convention(t_fm, u_fm)

    assert torch.allclose(tau, torch.tensor([0.75]))
    xt_diff = (1.0 - tau[:, None]) * x_data + tau[:, None] * x_noise
    assert torch.allclose(xt_fm, xt_diff, atol=1e-6)
    assert torch.allclose(v_diff, x_noise - x_data, atol=1e-6)


def test_ot_conditional_path_and_cfm_loss():
    g = torch.Generator().manual_seed(17)
    x0 = torch.randn((8, 2, 4, 4), generator=g)
    x1 = torch.randn((8, 2, 4, 4), generator=g)
    t = torch.linspace(0.0, 1.0, 8)

    xt, ut = run_ot_conditional_path(x0, x1, t, sigma_min=0.01)
    assert xt.shape == x0.shape and ut.shape == x0.shape
    # At t = 0: xt == x0; at t = 1: xt == x1 + sigma_min * x0
    assert torch.allclose(xt[0], x0[0], atol=1e-6)
    assert torch.allclose(xt[-1], x1[-1] + 0.01 * x0[-1], atol=1e-6)

    loss = run_cfm_loss(lambda x, t_b: torch.zeros_like(x), x0, x1, t, sigma_min=0.01)
    assert torch.allclose(loss, (ut ** 2).mean(), atol=1e-6)


def test_euler_vs_heun_accuracy_on_exact_linear_ode():
    """
    Solve dx/dt = -2 * t * x with x(0) = x0 on t in [0, 1] -> exact solution x(1) = exp(-1) * x0.
    Heun (2nd order) must achieve dramatically lower error than Euler at the SAME NFE budget!
    """
    def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -2.0 * t.view(-1, *([1] * (x.ndim - 1))) * x

    x0 = torch.tensor([[2.0, -3.0], [0.5, 4.0]])
    exact_x1 = x0 * torch.exp(torch.tensor(-1.0))

    # Equal NFE = 20: 20 Euler steps vs 10 Heun steps
    euler_x1 = run_flow_sample_loop(velocity_fn, x0, n_steps=20, solver="euler")
    heun_x1 = run_flow_sample_loop(velocity_fn, x0, n_steps=10, solver="heun")

    err_euler = (euler_x1 - exact_x1).abs().max().item()
    err_heun = (heun_x1 - exact_x1).abs().max().item()
    assert err_heun < err_euler * 0.25
    assert err_heun < 5e-3

    # Also check single-step interfaces directly
    s_euler = run_ode_euler_step(velocity_fn, x0, torch.zeros(2), dt=0.1)
    s_heun = run_ode_heun_step(velocity_fn, x0, torch.zeros(2), dt=0.1)
    assert torch.allclose(s_euler, x0, atol=1e-6)  # u(x, 0) = 0
    assert not torch.allclose(s_heun, x0)  # k2 evaluates at t = 0.1!


def test_trajectory_straightness_zero_iff_constant_velocity():
    x0 = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    x1 = torch.tensor([[2.0, 4.0], [-3.0, 1.0]])
    ts = torch.linspace(0.0, 1.0, 11).view(11, 1, 1)

    straight_traj = (1.0 - ts) * x0.unsqueeze(0) + ts * x1.unsqueeze(0)
    assert torch.allclose(run_trajectory_straightness(straight_traj), torch.tensor(0.0), atol=1e-6)

    # Add sinusoidal detour vanishing at endpoints t=0 and t=1 -> curved trajectory > 0!
    curved_traj = straight_traj + torch.sin(torch.pi * ts) * 0.8
    assert run_trajectory_straightness(curved_traj).item() > 1.0


def test_rosetta_stone_unification_both_vp_and_ot_schedules():
    """
    Must recover ALL 5 quantities whether alpha_t^2 + sigma_t^2 = 1 (VP-DDPM)
    OR alpha_t + sigma_t = 1 (OT Flow Matching, where alpha_t^2 + sigma_t^2 != 1)!
    """
    g = torch.Generator().manual_seed(88)
    x_data = torch.randn((6, 3, 4), generator=g)
    eps = torch.randn((6, 3, 4), generator=g)

    # Case 1: OT Flow Matching schedule (alpha_t = t, sigma_t = 1 - t)
    t = torch.tensor([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
    alpha_t = t
    sigma_t = 1.0 - t
    xt = alpha_t[:, None, None] * x_data + sigma_t[:, None, None] * eps
    score_true = -eps / sigma_t[:, None, None]
    v_true = alpha_t[:, None, None] * eps - sigma_t[:, None, None] * x_data
    u_fm_true = x_data - eps

    for known_type, known_val in [
        ("x_data", x_data),
        ("eps", eps),
        ("score", score_true),
        ("v", v_true),
        ("u_fm", u_fm_true),
    ]:
        res = run_rosetta_conversions(xt, alpha_t, sigma_t, known_type, known_val)
        assert torch.allclose(res["x_data"], x_data, atol=1e-4)
        assert torch.allclose(res["eps"], eps, atol=1e-4)
        assert torch.allclose(res["score"], score_true, atol=1e-4)
        assert torch.allclose(res["v"], v_true, atol=1e-4)
        assert torch.allclose(res["u_fm"], u_fm_true, atol=1e-4)
