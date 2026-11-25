# Assignment 5 (Lecture 5): Image Generation Architectures — U-Net, DiT, MM-DiT, & FLOPs Accounting

**Course:** Stanford CME 296 — Diffusion & Large Vision Models (Custom CS336-Style Curriculum)  
**Total Points:** 50 points (16 pts Written/Brainstorming · 26 pts Code via `pytest` · 8 pts Experiments)

---

## 1. Assignment Overview

Lecture 5 traces the architectural shift at the heart of modern visual generation: from convolutional **U-Nets** (DDPM, ADM, SD 1.5 / SDXL) to **Diffusion Transformers (DiT)** (Peebles & Xie, Sora) and **Multimodal Diffusion Transformers (MM-DiT)** (Stable Diffusion 3, FLUX). Just like Stanford CS336 Assignment 1 §3.6, you will build every architectural primitive from scratch **and** perform exact parameter & FLOPs resource accounting across scales and image resolutions!

### What you will implement
1. Sinusoidal timestep embeddings (`run_sinusoidal_timestep_embedding`)
2. FiLM-conditioned convolutional U-Net `ResBlock` with GroupNorm + SiLU + $1\times 1$ channel shortcut (`run_unet_resblock_film`)
3. Lossless 2D `patchify` & `unpatchify` (`run_patchify`, `run_unpatchify`)
4. **2D Axial Rotary Position Embeddings (2D RoPE)** (`run_2d_axial_rope`)
5. **`adaLN-Zero` DiT Block** (`run_adaln_zero_block`)
6. **MM-DiT Double-Stream Joint-Attention Block** (`run_mmdit_block`)
7. **CS336-Style Exact DiT FLOPs Accounting** (`run_dit_flops_accounting`)

---

## 2. Convolutional U-Nets & Conditioning Primitives

### Problem (`a5_unet_skip_brainstorm`): Skip-Connection Variance & Inductive Biases (4 points)
**(a)** In a U-Net with $K$ encoder–decoder skip additions $h_{\text{dec}}^{(k)} \leftarrow h_{\text{dec}}^{(k)} + h_{\text{enc}}^{(k)}$, suppose each encoder feature map and decoder feature map initially have independent zero-mean unit variance ($\operatorname{Var} = 1$). What happens to activation variance after $K = 12$ skip additions if unscaled, and what constant factor restores unit variance at every skip merge?  
**(b)** Compare a convolutional U-Net against an isotropic patchified `DiT` on two axes: **(i)** spatial translation equivariance at initialization, and **(ii)** why `DiT` scales much more cleanly with compute ($G\text{FLOPs}$) despite lacking built-in multi-scale inductive bias.

### Problem (`a5_unet_code`): Implement Sinusoidal Embeddings & FiLM U-Net ResBlock (7 points)
Implement `[adapters.run_sinusoidal_timestep_embedding]` and `[adapters.run_unet_resblock_film]` (implementing GroupNorm and SiLU from basic tensor ops — do not call `F.group_norm` or `F.silu`!), then run:
```bash
pytest tests/test_a5_architectures.py -k "sinusoidal or unet_resblock"
```

---

## 3. Diffusion Transformers (DiT): Patching, 2D Axial RoPE, & `adaLN-Zero`

### Problem (`a5_adaln_zero_brainstorm`): Why Zero-Initializing `adaLN` Beats Standard Conditioning (4 points)
In `DiT`, each block predicts 6 vectors $(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) = W_{\text{adaLN}}\operatorname{SiLU}(c) + b_{\text{adaLN}}$ and computes:
$$x' = x + \alpha_1 \odot \operatorname{MHA}\!\big((1 + \gamma_1) \odot \operatorname{LN}(x) + \beta_1\big), \qquad x'' = x' + \alpha_2 \odot \operatorname{MLP}\!\big((1 + \gamma_2) \odot \operatorname{LN}(x') + \beta_2\big)$$
with $W_{\text{adaLN}} = 0, b_{\text{adaLN}} = 0$, **and** the final unpatchify linear projection also initialized to zero.  
**(a)** What is the exact forward output of a 28-layer `DiT-XL/2` at training step 0, and what is the Jacobian $\frac{\partial x^{(L)}}{\partial x^{(0)}}$ across the 28 residual blocks at step 0?  
**(b) Brainstorm:** Wait — if $W_{\text{adaLN}} = 0$ makes $\alpha_1 = \alpha_2 = 0$ at step 0, why don't all gradients into `MHA` and `MLP` stay permanently stuck at zero forever?! Trace $\frac{\partial \mathcal{L}}{\partial W_{\text{adaLN}}}$ vs. $\frac{\partial \mathcal{L}}{\partial W_{\text{qkv}}}$ on step 1 vs. step 2!

### Problem (`a5_dit_primitives`): Implement Patchify, 2D Axial RoPE, & `adaLN-Zero` Block (11 points)
1. Implement `[adapters.run_patchify]` and `[adapters.run_unpatchify]`.
2. Implement `[adapters.run_2d_axial_rope]`, splitting head dimension $d_h$ into row-frequency rotations (`d_h // 2`) and column-frequency rotations (`d_h // 2`). Notice in `test_2d_axial_rope_norm_and_2d_relative_invariance` how $\langle \operatorname{RoPE}(q, y_1, x_1), \operatorname{RoPE}(k, y_2, x_2)\rangle$ depends **purely on the 2D offset $(y_1 - y_2, x_1 - x_2)$** — enabling variable aspect-ratio and multi-resolution generation!
3. Implement `[adapters.run_adaln_zero_block]`.

Run:
```bash
pytest tests/test_a5_architectures.py -k "patchify or rope or adaln_zero"
```

---

## 4. Multimodal Diffusion Transformer (MM-DiT — SD3 & FLUX)

### Problem (`a5_mmdit_brainstorm`): Cross-Attention vs. MM-DiT Joint Double-Stream Attention (4 points)
Classical text-to-image models (SD 1.5 / SDXL / PixArt) inject text via **cross-attention** ($Q$ from image tokens $X \in \mathbb{R}^{N_x \times D}$, $K, V$ from frozen text embeddings $C \in \mathbb{R}^{N_c \times D}$), whereas **MM-DiT** (Esser et al., 2024; FLUX) maintains two parallel residual streams $(X, C)$ with **separate** modality-specific `adaLN-Zero`, $QKV$, output, and MLP weights, merging them only inside **concatenated self-attention** over $[Q_x; Q_c], [K_x; K_c], [V_x; V_c] \in \mathbb{R}^{(N_x + N_c) \times D}$.  
**(a)** Break the joint $(N_x + N_c) \times (N_x + N_c)$ attention matrix into **four quadrants** ($X\leftarrow X$, $X\leftarrow C$, $C\leftarrow X$, $C\leftarrow C$) and explain what new information pathways exist in MM-DiT that are completely absent in frozen cross-attention!  
**(b)** Why would sharing a single $W_{\text{qkv}}$ and `MLP` across both image VAE patches and T5/CLIP text tokens hurt performance compared to MM-DiT's separate stream weights?

### Problem (`a5_mmdit_code`): Implement MM-DiT Double-Stream Block (5 points)
Implement `[adapters.run_mmdit_block]` and test:
```bash
pytest tests/test_a5_architectures.py -k mmdit
```

---

## 5. CS336-Style Resource & FLOPs Accounting

Recall the CS336 matrix-multiply FLOPs rule: multiplying $(M \times K)$ by $(K \times N)$ requires **$2 M N K$ FLOPs**.

### Problem (`a5_dit_flops_accounting`): Parameter, FLOPs, & High-Res Scaling Analysis (7 points: 4 Written + 3 Code)
**(a) Code:** Implement `[adapters.run_dit_flops_accounting]` and pass the entire Assignment 5 suite:
```bash
pytest tests/test_a5_architectures.py
```
**(b) Written Accounting (`DiT-S/2`, `DiT-B/2`, `DiT-XL/2`):**  
Using your `run_dit_flops_accounting` function (`batch_size=1, in_channels=4, learn_sigma=True`):
1. Report total forward GFLOPs and component percentages (`attn_proj`, `attn_scores`, `mlp`, `adaln`) for **`DiT-XL/2`** ($L=28, D=1152, H=16$) across three settings:
   - $256\times 256$ image ($32\times 32$ latent), patch size $p = 4$ ($N = 64$)
   - $256\times 256$ image ($32\times 32$ latent), patch size $p = 2$ ($N = 256$)
   - $1024\times 1024$ image ($128\times 128$ latent), patch size $p = 2$ ($N = 4096$)
2. Derive algebraically the **exact crossover token count $N^*$** (as a function of $D$ for `mlp_ratio = 4`) at which quadratic attention score FLOPs ($Q K^\top$ and $A V$) surpass feed-forward `MLP` FLOPs!
3. During Classifier-Free Guidance inference (`2B` forward pass), how much memory (in MB, `bf16`) do materialized $(2B, H, N, N)$ attention matrices consume for a single layer of `DiT-XL/2` at $2048\times 2048$ resolution ($256\times 256$ latent $\implies N = 16{,}384$, $B = 4$), and how does **FlashAttention-2** eliminate that bottleneck?

---

## 6. Capstone Head-to-Head Experiment: Tiny U-Net vs. Tiny DiT vs. Tiny MM-DiT

### Problem (`a5_arch_head_to_head`): Matched-Parameter Flow Matching Benchmark (8 points)
Assemble a ~300K-parameter **Tiny U-Net**, **Tiny DiT** (`patch_size=2`, `adaLN-Zero` + 2D RoPE), and **Tiny MM-DiT** (representing the class condition as a 2-token prompt sequence `[BOS, class_tok]`) in `cme296_assignments/` and train all three with Conditional Flow Matching (`3,000` steps, `batch_size=128`) on `sample_procedural_shapes(image_size=16)`:
1. Verify empirically that initializing `adaLN` weights to `0` (`adaLN-Zero`) converges faster than random normal `adaLN` initialization (`std=0.02`)!
2. Compare training loss curves, wall-clock step time, forward FLOPs, and 20-step Heun generated sample grids across U-Net, DiT, and MM-DiT.
