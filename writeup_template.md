# CME 296 Custom Assignments — Writeup Template

Fill in your derivations, brainstorming answers, FLOPs tables, and experiment plots below (similar to CS336's `writeup.pdf`).

---

## Assignment 1: Discrete-Time Diffusion (Lecture 1 — 50 pts)

### `a1_markov_jump` (4 pts)
- **(a) Induction proof of $q(x_t \mid x_0)$:**  
- **(b) Variance Preservation ($\operatorname{Cov}(x_t)$ when $\operatorname{Cov}(x_0)=I$):**  
- **(c) Variance trap on $(1-\lambda_t)x_0 + \lambda_t \epsilon$:**  

### `a1_resolution_snr_brainstorm` (4 pts)
- **(a) Effective SNR of $p\times p$ block-averaged pixels:**  
- **(b) Required log-SNR shift when scaling resolution by $p\times$:**  

### `a1_elbo_derivation` (6 pts)
- **(a) ELBO decomposition ($L_T + \sum L_{t-1} + L_0$):**  
- **(b) Why conditioning reverse transition on $x_0$ is essential:**  
- **(c) Bayes completion of square for $\tilde{\mu}_t$ and $\tilde{\beta}_t$:**  

### `a1_v_pred_brainstorm` (3 pts)
- **(a) Effective $x_0$-space SNR weights of $\epsilon$ and $v$ losses:**  
- **(b) High-noise conditioning of $\epsilon$-pred vs. $v$-pred:**  

### `a1_ddim_brainstorm` (4 pts)
- **(a) Marginal preservation proof for arbitrary $\sigma_t$:**  
- **(b) Why DDIM needs zero retraining:**  

### `a1_2d_experiment` (8 pts)
- **Checkerboard accuracy table across NFE $\in \{10, 25, 50, 200, 1000\}$ (DDPM vs. DDIM):**  
- **`slerp` vs. `lerp` interpolation plot & norm explanation:**  

---

## Assignment 2: Score Matching & SDEs (Lecture 2 — 50 pts)

### `a2_ism_dsm_proof` (6 pts)
- **(a) Why $\nabla_x \log p_\theta(x)$ kills $Z_\theta$ (unlike $\nabla_\theta \log p_\theta(x)$):**  
- **(b) Hyvärinen integration-by-parts proof + Hutchinson $O(1)$ VJP & Rademacher variance:**  
- **(c) Vincent's DSM gradient equivalence proof:**  

### `a2_low_density_brainstorm` (4 pts)
- **(a) Analytical score near $\mu_1$ (cancellation of $\pi_1$):**  
- **(b) Why unannealed Langevin misses `80/20` weights & how ALD fixes it:**  

### `a2_sde_and_fokker_planck` (8 pts)
- **(a) Continuum $\Delta t \to 0$ VP-SDE derivation & closed-form perturbation kernel:**  
- **(b) Fokker-Planck $\to$ Continuity Equation proof of PF-ODE ($\frac{1}{2}$ factor):**  

### `a2_mode_proportions_experiment` (8 pts)
- **Recovered Mode-1 mass table (Fixed-$\sigma$ Langevin vs. ALD vs. Reverse SDE vs. PF-ODE):**  
- **Quiver plots & analysis:**  

---

## Assignment 3: Flow Matching & Rectified Flow (Lecture 3 — 50 pts)

### `a3_cnf_bottleneck` (4 pts)
- **(a) Discrete NF Jacobian determinant $O(d^3)$ bottleneck:**  
- **(b) Why maximum-likelihood CNF training was still slow:**  

### `a3_convention_bridge` (2 pts written)
- **Chain-rule relation between $(t, u_t)$ and $(\tau, v_\tau)$:**  

### `a3_cfm_gradient_proof` (5 pts)
- **Law-of-total-expectation proof that $\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$:**  

### `a3_rectified_flow_brainstorm` (5 pts)
- **(a) Crossing chords example & why 1-RF ODE paths curve:**  
- **(b) Why Reflow uncrosses couplings & $S(Z)=0 \implies$ exact 1-step Euler:**  

### `a3_rosetta_stone` (2 pts written)
- **General $2\times 2$ matrix inversions for $(x_t, v)$ and $(x_t, u_{\text{FM}})$:**  

### `a3_reflow_2d_experiment` (8 pts)
- **1-RF vs. 2-RF straightness & 1-step/2-step Euler comparison plots:**  

---

## Assignment 4: Latent Space, Contrastive Learning, & Guidance (Lecture 4 — 50 pts)

### `a4_vae_elbo_and_scaling` (6 pts)
- **(a) VAE ELBO & closed-form diagonal Gaussian KL derivation:**  
- **(b) `0.18215` unscaled-latent SNR distortion & test-time mismatch analysis:**  

### `a4_clip_vs_siglip_brainstorm` (5 pts)
- **(a) Distributed all-gather bottleneck in CLIP vs. SigLIP:**  
- **(b) Step-0 negative/positive gradient imbalance at $B = 32{,}768$ ($b=0$ vs. $b=-10$):**  

### `a4_guidance_brainstorm` (6 pts)
- **(a) Tilted posterior $\tilde{p}_\gamma(x_t\mid c) \propto p(x_t)p(c\mid x_t)^\gamma$ & CFG equivalence:**  
- **(b) Two failure modes of Classifier Guidance:**  
- **(c) Variance blowup calculation at $w = 10, \rho = 0.85$ & Std-Rescaling:**  

### `a4_latent_cfg_experiment` (8 pts)
- **`LatentScaler` ablation + CFG quality–diversity trade-off curves & grids:**  

---

## Assignment 5: U-Net, DiT, MM-DiT, & FLOPs Accounting (Lecture 5 — 50 pts)

### `a5_unet_skip_brainstorm` (4 pts)
- **(a) $1/\sqrt{2}$ skip variance stabilization:**  
- **(b) U-Net vs. DiT inductive bias & compute scaling:**  

### `a5_adaln_zero_brainstorm` (4 pts)
- **(a) Step-0 output & Jacobian of 28-layer `DiT-XL/2`:**  
- **(b) Two-step gradient unblocking trace ($\nabla_{W_{\text{adaLN}}}$ on step 1 $\to \nabla_{W_{\text{qkv}}}$ on step 2):**  

### `a5_mmdit_brainstorm` (4 pts)
- **(a) Four quadrants of MM-DiT joint attention vs. cross-attention:**  
- **(b) Why separate modality stream projections matter:**  

### `a5_dit_flops_accounting` (4 pts written)
- **1. `DiT-XL/2` GFLOPs & component % table ($256\text{px } p=4$, $256\text{px } p=2$, $1024\text{px } p=2$):**  
- **2. Exact crossover token count $N^*$ where attention scores overtake MLP:**  
- **3. `2048px` `bf16` attention matrix memory & FlashAttention-2 analysis:**  

### `a5_arch_head_to_head` (8 pts)
- **Tiny U-Net vs. Tiny DiT (`adaLN-Zero` vs. random init) vs. Tiny MM-DiT comparison:**  
