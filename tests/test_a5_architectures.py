from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.adapters import (
    run_2d_axial_rope,
    run_adaln_zero_block,
    run_dit_flops_accounting,
    run_mmdit_block,
    run_patchify,
    run_sinusoidal_timestep_embedding,
    run_unpatchify,
    run_unet_resblock_film,
)


def test_sinusoidal_timestep_embedding_even_and_odd():
    t = torch.tensor([0.0, 1.0, 500.0])
    emb_even = run_sinusoidal_timestep_embedding(t, dim=16)
    assert emb_even.shape == (3, 16)
    # At t = 0: cos(0) = 1 for first half, sin(0) = 0 for second half
    assert torch.allclose(emb_even[0, :8], torch.ones(8), atol=1e-6)
    assert torch.allclose(emb_even[0, 8:], torch.zeros(8), atol=1e-6)

    emb_odd = run_sinusoidal_timestep_embedding(t, dim=17)
    assert emb_odd.shape == (3, 17)
    assert torch.allclose(emb_odd[:, :16], emb_even, atol=1e-6)
    assert torch.allclose(emb_odd[:, 16], torch.zeros(3), atol=1e-7)


def test_patchify_unpatchify_exact_roundtrip():
    g = torch.Generator().manual_seed(9)
    imgs = torch.randn((3, 4, 16, 12), generator=g)
    patches = run_patchify(imgs, patch_size=4)
    assert patches.shape == (3, 12, 4 * 4 * 4)

    # Verify exact spatial layout on top-left patch (row=0, col=0) -> ordered (p_h, p_w, C)
    top_left = imgs[:, :, :4, :4].permute(0, 2, 3, 1).reshape(3, -1)
    assert torch.allclose(patches[:, 0, :], top_left, atol=1e-6)

    recon = run_unpatchify(patches, patch_size=4, channels=4, height=16, width=12)
    assert torch.allclose(recon, imgs, atol=1e-6)


def test_2d_axial_rope_norm_and_2d_relative_invariance():
    """
    Key property of 2D Axial RoPE:
      1. Norm preserving: ||RoPE(q)|| == ||q|| at every token.
      2. 2D translation invariance: <RoPE(q, y1, x1), RoPE(k, y2, x2)> depends ONLY on
         relative displacement (y1 - y2, x1 - x2)!
    """
    g = torch.Generator().manual_seed(296)
    B, H, gh, gw, dh = 2, 4, 4, 5, 16
    x = torch.randn((B, H, gh * gw, dh), generator=g)
    rx = run_2d_axial_rope(x, grid_h=gh, grid_w=gw)
    assert rx.shape == x.shape
    assert torch.allclose(rx.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

    # Put identical query q0 and key k0 everywhere on the 4x5 grid and check relative dot products
    q_vec = torch.randn((1, 1, 1, dh), generator=g).expand(1, 1, gh * gw, dh).clone()
    k_vec = torch.randn((1, 1, 1, dh), generator=g).expand(1, 1, gh * gw, dh).clone()
    rq = run_2d_axial_rope(q_vec, grid_h=gh, grid_w=gw)
    rk = run_2d_axial_rope(k_vec, grid_h=gh, grid_w=gw)

    def idx(r: int, c: int) -> int:
        return r * gw + c

    # Displacement (dy=+1, dx=+2) from (0, 0)->(1, 2) must match (2, 1)->(3, 3)!
    dot_a = (rq[0, 0, idx(1, 2)] * rk[0, 0, idx(0, 0)]).sum()
    dot_b = (rq[0, 0, idx(3, 3)] * rk[0, 0, idx(2, 1)]).sum()
    assert torch.allclose(dot_a, dot_b, atol=1e-5)


def test_unet_resblock_film_channel_projection():
    g = torch.Generator().manual_seed(12)
    B, C_in, C_out, D_cond = 2, 8, 16, 12
    x = torch.randn((B, C_in, 8, 8), generator=g)
    cond = torch.randn((B, D_cond), generator=g)
    weights = {
        "gn1_weight": torch.ones(C_in),
        "gn1_bias": torch.zeros(C_in),
        "conv1_weight": torch.randn((C_out, C_in, 3, 3), generator=g) * 0.1,
        "conv1_bias": torch.zeros(C_out),
        "film_weight": torch.randn((2 * C_out, D_cond), generator=g) * 0.1,
        "film_bias": torch.zeros(2 * C_out),
        "gn2_weight": torch.ones(C_out),
        "gn2_bias": torch.zeros(C_out),
        "conv2_weight": torch.randn((C_out, C_out, 3, 3), generator=g) * 0.1,
        "conv2_bias": torch.zeros(C_out),
        "skip_weight": torch.randn((C_out, C_in, 1, 1), generator=g) * 0.1,
        "skip_bias": torch.zeros(C_out),
    }
    out = run_unet_resblock_film(x, cond, weights, num_groups=4)
    assert out.shape == (B, C_out, 8, 8)
    # Zeroing conv2 makes residual branch zero -> output must equal 1x1 skip shortcut!
    weights_zero = {**weights, "conv2_weight": torch.zeros_like(weights["conv2_weight"])}
    out_skip = run_unet_resblock_film(x, cond, weights_zero, num_groups=4)
    expected_skip = F.conv2d(x, weights["skip_weight"], weights["skip_bias"])
    assert torch.allclose(out_skip, expected_skip, atol=1e-6)


def test_adaln_zero_identity_at_init_and_nonzero_step():
    g = torch.Generator().manual_seed(33)
    B, N, D, H = 2, 9, 16, 4
    x = torch.randn((B, N, D), generator=g)
    cond = torch.randn((B, D), generator=g)

    weights = {
        "adaln_weight": torch.zeros((6 * D, D)),
        "adaln_bias": torch.zeros(6 * D),
        "qkv_weight": torch.randn((3 * D, D), generator=g) * 0.1,
        "qkv_bias": torch.zeros(3 * D),
        "proj_weight": torch.randn((D, D), generator=g) * 0.1,
        "proj_bias": torch.zeros(D),
        "fc1_weight": torch.randn((4 * D, D), generator=g) * 0.1,
        "fc1_bias": torch.zeros(4 * D),
        "fc2_weight": torch.randn((D, 4 * D), generator=g) * 0.1,
        "fc2_bias": torch.zeros(D),
    }
    # Step-0 adaLN-Zero guarantee: zero-initialized modulation linear layer -> exact identity!
    out_init = run_adaln_zero_block(x, cond, weights, num_heads=H)
    assert torch.allclose(out_init, x, atol=1e-7)

    # Non-zero modulation activates both attention and MLP residual branches
    weights["adaln_weight"] = torch.randn((6 * D, D), generator=g) * 0.2
    out_active = run_adaln_zero_block(x, cond, weights, num_heads=H)
    assert out_active.shape == x.shape
    assert not torch.allclose(out_active, x)


def test_mmdit_joint_bidirectional_attention():
    """
    Verify MM-DiT couples image and text streams bidirectionally: changing `txt_tokens`
    must change `img_out` AND changing `img_tokens` must change `txt_out`!
    """
    g = torch.Generator().manual_seed(44)
    B, N_img, N_txt, D, H = 2, 8, 5, 16, 4
    img = torch.randn((B, N_img, D), generator=g)
    txt1 = torch.randn((B, N_txt, D), generator=g)
    txt2 = torch.randn((B, N_txt, D), generator=g)
    cond = torch.randn((B, D), generator=g)

    weights = {}
    for prefix in ("img", "txt"):
        weights[f"{prefix}_adaln_weight"] = torch.randn((6 * D, D), generator=g) * 0.3
        weights[f"{prefix}_adaln_bias"] = torch.ones(6 * D) * 0.2
        weights[f"{prefix}_qkv_weight"] = torch.randn((3 * D, D), generator=g) * 0.2
        weights[f"{prefix}_qkv_bias"] = torch.zeros(3 * D)
        weights[f"{prefix}_proj_weight"] = torch.randn((D, D), generator=g) * 0.2
        weights[f"{prefix}_proj_bias"] = torch.zeros(D)
        weights[f"{prefix}_fc1_weight"] = torch.randn((4 * D, D), generator=g) * 0.2
        weights[f"{prefix}_fc1_bias"] = torch.zeros(4 * D)
        weights[f"{prefix}_fc2_weight"] = torch.randn((D, 4 * D), generator=g) * 0.2
        weights[f"{prefix}_fc2_bias"] = torch.zeros(D)

    img_out1, txt_out1 = run_mmdit_block(img, txt1, cond, weights, num_heads=H)
    img_out2, _ = run_mmdit_block(img, txt2, cond, weights, num_heads=H)
    assert img_out1.shape == img.shape and txt_out1.shape == txt1.shape
    assert not torch.allclose(img_out1, img_out2)

    img_alt = torch.randn((B, N_img, D), generator=g)
    _, txt_out_alt = run_mmdit_block(img_alt, txt1, cond, weights, num_heads=H)
    assert not torch.allclose(txt_out1, txt_out_alt)


def test_dit_xl2_flops_accounting_scaling():
    """Verify exact CS336-style FLOPs breakdown on DiT-XL/2 at 256px (32x32 latent) vs 1024px (128x128 latent)."""
    xl_256 = run_dit_flops_accounting(
        batch_size=1, in_channels=4, latent_h=32, latent_w=32,
        patch_size=2, num_layers=28, d_model=1152, num_heads=16, mlp_ratio=4.0, learn_sigma=True,
    )
    assert xl_256["num_tokens"] == 256
    # At N=256 (256x256 image), MLP FLOPs dominate quadratic attention score FLOPs
    assert xl_256["mlp_flops"] > 5 * xl_256["attn_scores_flops"]
    assert xl_256["total_flops"] == (
        xl_256["patch_embed_flops"]
        + xl_256["adaln_flops"]
        + xl_256["attn_proj_flops"]
        + xl_256["attn_scores_flops"]
        + xl_256["mlp_flops"]
        + xl_256["final_head_flops"]
    )

    # At 1024x1024 image (128x128 latent -> N=4096 tokens, 16x more tokens):
    # MLP scales 16x, while attention QK^T/AV scales 256x (reaching 4096/4608 = 88.9% of MLP on XL/2,
    # and overtaking MLP whenever N > 4 * d_model, e.g. on DiT-B/2 where 4 * 768 = 3072 < 4096!)
    xl_1024 = run_dit_flops_accounting(
        batch_size=1, in_channels=4, latent_h=128, latent_w=128,
        patch_size=2, num_layers=28, d_model=1152, num_heads=16, mlp_ratio=4.0, learn_sigma=True,
    )
    assert xl_1024["num_tokens"] == 4096
    assert xl_1024["mlp_flops"] == 16 * xl_256["mlp_flops"]
    assert xl_1024["attn_scores_flops"] == 256 * xl_256["attn_scores_flops"]

    b_1024 = run_dit_flops_accounting(
        batch_size=1, in_channels=4, latent_h=128, latent_w=128,
        patch_size=2, num_layers=12, d_model=768, num_heads=12, mlp_ratio=4.0, learn_sigma=True,
    )
    assert b_1024["attn_scores_flops"] > b_1024["mlp_flops"]
