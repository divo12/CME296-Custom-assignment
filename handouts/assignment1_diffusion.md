# Assignment 1 (Lecture 1): Discrete-Time Diffusion — DDPM, ELBO, Schedules, and DDIM

**Course:** Stanford CME 296 — Diffusion & Large Vision Models (Custom CS336-Style Curriculum)  
**Total Points:** 50 points (17 pts Written/Brainstorming · 25 pts Code via `pytest` · 8 pts Experiments)

---

## 1. Assignment Overview

In Lecture 1, you studied how **Denoising Diffusion Probabilistic Models (DDPM)** turn generative modeling into a sequence of small denoising regressions by reversing a fixed Markov chain. In this assignment, you will derive the variational foundations by hand, build every discrete-time diffusion primitive from scratch, prove why $\epsilon$, $x_0$, and $v$ prediction become equivalent under SNR weighting, and implement non-Markovian **DDIM** strided sampling.

### What you will implement

1. Variance-Preserving (VP) linear and cosine noise schedules + log-SNR (`run_noise_schedule`)
2. Closed-form Markov jump $q(x_t | x_0)$ (`run_ddpm_forward`)
3. Exact Bayes reverse posterior $q(x_{t-1} | x_t, x_0)$ (`run_ddpm_posterior`)
4. $(\epsilon, x_0, v)$ parameterization conversions and Min-SNR $\gamma$ weighted training loss (`run_convert_predictions`, `run_ddpm_loss`)
5. Ancestral DDPM reverse sampler and strided deterministic/stochastic DDIM sampler (`run_ddpm_step`, `run_ddim_step`)

### What you can use

Build every diffusion primitive from scratch using basic `torch` tensor operations (`torch.sqrt`, `torch.cumprod`, `torch.cos`, `torch.where`, etc.) and `torch.nn.Module` / `torch.nn.Parameter` for your experiment network. You may **not** import `diffusers` or external diffusion schedule/sampler packages.

> **CME 296 Time Convention (Lecture 1):**  
> Clean data lives at $t = 0$ ($x_0 \sim p_{\text{data}}$) and pure Gaussian noise lives at $t = T$ ($x_T \approx \mathcal{N}(0, I)$).  
> **Code indexing note:** In PyTorch schedule tensors of length `T`, 0-indexed `t_idx = 0` corresponds to paper step $t = 1$ (least noisy latent) and `t_idx = T - 1` corresponds to paper step $t = T$ (most noisy latent). At `t_idx == 0`, the previous clean state has $\bar{\alpha}_{\text{prev}} = 1.0$.

> **Low-Resource Tip (CPU / Apple Silicon MPS):**  
> Section 5 trains a 4-layer MLP denoiser (~50K parameters) on the provided 2D `sample_2d_checkerboard` and `sample_2d_swiss_roll` distributions in `cme296_datasets`. Full training (5,000 steps, batch size 256) takes **under 15 seconds on CPU** and **under 5 seconds on MPS/CUDA**.

---

## 2. Forward Process and Noise Schedules

Given schedule $\beta_1, \dots, \beta_T \in (0, 1)$, define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. A single Markov step is:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\; \sqrt{1 - \beta_t}\,x_{t-1},\; \beta_t I\right)
$$

### Problem (`a1_markov_jump`): Closed-Form Jump and Variance Preservation (4 points)

**(a)** Prove by induction (using reparameterization and the rule for summing independent Gaussians) that:

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t;\; \sqrt{\bar{\alpha}_t}\,x_0,\; (1 - \bar{\alpha}_t) I\right)
$$

**(b)** Suppose clean data is standardized so $\mathbb{E}[x_0] = 0$ and $\text{Cov}(x_0) = I$. Compute the marginal covariance $\text{Cov}(x_t)$ integrated over $x_0 \sim p_{\text{data}}$ and explain in one sentence why Ho et al.'s chain is called **Variance Preserving (VP)**.

**(c) Brainstorm:** Suppose a classmate mistakenly writes $x_t = (1 - \lambda_t) x_0 + \lambda_t \epsilon$ with $\lambda_t \in [0, 1]$ and claims $\text{Var}(x_t) = 1$ everywhere because $(1 - \lambda_t) + \lambda_t = 1$. Compute the actual variance at $\lambda_t = 0.5$ and state the minimum value it hits.

**Deliverable:** Step-by-step derivation for (a)–(c) in `writeup.md` or `writeup.pdf`.

### Problem (`a1_schedules`): Implement Linear and Cosine Noise Schedules (5 points)

Implement both schedules of length $T$:

- **Linear schedule** (Ho et al., 2020): $\beta_t$ linearly spaced from $\beta_{\text{start}} = 10^{-4}$ to $\beta_{\text{end}} = 0.02$.
- **Cosine schedule** (Nichol & Dhariwal, 2021):

$$
f(k) = \cos^2\left(\frac{k / T + s}{1 + s} \cdot \frac{\pi}{2}\right), \qquad \tilde{\alpha}_k = \frac{f(k)}{f(0)} \quad (k = 0, \dots, T)
$$

$$
\beta_k = \min\left(1 - \frac{\tilde{\alpha}_k}{\tilde{\alpha}_{k-1}},\; 0.999\right) \quad (k = 1, \dots, T)
$$

Then compute $\alpha_k = 1 - \beta_k$, $\bar{\alpha}_k = \prod_{j=1}^k \alpha_j$, and the log signal-to-noise ratio $\lambda_k = \log\left(\frac{\bar{\alpha}_k}{1 - \bar{\alpha}_k}\right)$.

Wire your implementation to `[adapters.run_noise_schedule]` and `[adapters.run_ddpm_forward]`, then run:

```bash
pytest tests/test_a1_diffusion.py -k "schedules or forward"
```

**Deliverable:** Passing unit tests.

### Problem (`a1_resolution_snr_brainstorm`): Why Resolution Breaks Noise Schedules (4 points)

A schedule tuned on $64 \times 64$ images often fails badly when trained directly on $512 \times 512$ images at the exact same $\bar{\alpha}_t$ curve — high-res models spend almost all steps seeing clean global structure.

To see why mathematically, consider a flat image patch of $p \times p$ identical pixels ($x_{0,i} = c$ for all $i = 1, \dots, p^2$) corrupted via $x_{t,i} = \sqrt{\bar{\alpha}_t}\,c + \sqrt{1 - \bar{\alpha}_t}\,\epsilon_i$ with independent $\epsilon_i \sim \mathcal{N}(0, 1)$.

**(a)** Compute the **effective SNR** (ratio of squared signal mean to noise variance) of the $p \times p$ block-averaged pixel $\bar{x}_t = \frac{1}{p^2}\sum_{i=1}^{p^2} x_{t,i}$ in terms of $\text{SNR}_t = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$ and $p$.

**(b)** Derive how you must shift $\lambda_t = \log \text{SNR}_t$ when scaling image resolution from $H \times W$ to $(pH) \times (pW)$ so coarse low-frequency structure experiences the **same** effective SNR at every $t \in [0, T]$.

**Deliverable:** A 4–6 line derivation explaining why modern high-res models use **shifted cosine / shifted flow** schedules.

---

## 3. Variational Formulation (ELBO) and Tractable Reverse Posterior

### Problem (`a1_elbo_derivation`): ELBO Decomposition and Bayes Posterior (6 points)

**(a)** Starting from $-\log p_\theta(x_0) = -\log \int p_\theta(x_{0:T})\,\mathrm{d}x_{1:T}$, apply importance sampling with $q(x_{1:T} \mid x_0)$ and Jensen's inequality, then rewrite:

$$
q(x_{1:T} \mid x_0) = q(x_1 \mid x_0) \prod_{t=2}^T q(x_{t-1} \mid x_t, x_0)\,\frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}
$$

(a telescoping product) to obtain:

$$
\mathcal{L}_{\text{VLB}} = \underbrace{D_{\text{KL}}\left(q(x_T \mid x_0) \,\|\, p(x_T)\right)}_{L_T} + \sum_{t=2}^T \underbrace{D_{\text{KL}}\left(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\right)}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 \mid x_1)}_{-L_0}
$$

**(b) Brainstorm:** Why would expanding $q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})$ *without* conditioning the reverse transition on $x_0$ leave us stuck with an intractable KL divergence?

**(c)** Using Bayes' rule $q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)$, complete the quadratic form in $x_{t-1}$ to derive $\tilde{\mu}_t(x_t, x_0)$ and $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t$.

**Deliverable:** Clean algebraic derivation for (a)–(c).

### Problem (`a1_posterior`): Implement Reverse Posterior (5 points)

Implement `q_posterior(x0, xt, t_idx)` returning `(posterior_mean, posterior_var, posterior_log_var_clipped)` for $q(x_{t-1} \mid x_t, x_0)$. Remember to clip the log-variance at `t_idx == 0` to step `1`'s log-variance so $\log \tilde{\beta}_0$ never produces `-inf`.

Wire `[adapters.run_ddpm_posterior]` and run:

```bash
pytest tests/test_a1_diffusion.py -k test_ddpm_posterior_bayes_consistency
```

---

## 4. Parameterizations and SNR Weighting

Instead of predicting $\tilde{\mu}_t$, modern diffusion models predict $\epsilon_\theta$, $\hat{x}_{0,\theta}$, or Salimans & Ho's **velocity**:

$$
v = \sqrt{\bar{\alpha}_t}\,\epsilon - \sqrt{1 - \bar{\alpha}_t}\,x_0
$$

### Problem (`a1_v_pred_brainstorm`): High-Noise Conditioning and Min-SNR Equivalence (3 points)

**(a)** Express unweighted $\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2$ and $\|v - v_\theta(x_t, t)\|_2^2$ in terms of $\|x_0 - \hat{x}_{0,\theta}(x_t, t)\|_2^2$ and $\text{SNR}_t = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$.

**(b)** What happens to $\hat{x}_0 = (x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta) / \sqrt{\bar{\alpha}_t}$ as $t \to T$ ($\bar{\alpha}_t \to 0$) if $\epsilon_\theta$ makes a tiny $10^{-3}$ error? Why does $v$ prediction remain well-conditioned at **both** $t \to 0$ and $t \to T$?

**Deliverable:** Exact SNR weight expressions + 2–3 sentence explanation.

### Problem (`a1_predictions_and_loss`): Implement Conversions and Min-SNR Loss (8 points)

Implement `[adapters.run_convert_predictions]` and `[adapters.run_ddpm_loss]`. Verify that under `weight_type="min_snr"`, all three parameterizations produce **numerically identical** weighted losses:

```bash
pytest tests/test_a1_diffusion.py -k "conversions or min_snr"
```

---

## 5. Ancestral DDPM vs. Non-Markovian DDIM Sampling

### Problem (`a1_ddim_brainstorm`): Why DDIM Needs Zero Retraining (4 points)

Song et al. (2020) define a non-Markovian family indexed by $\sigma_t \ge 0$:

$$
q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}}\,x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\,\frac{x_t - \sqrt{\bar{\alpha}_t}\,x_0}{\sqrt{1 - \bar{\alpha}_t}},\; \sigma_t^2 I\right)
$$

**(a)** Prove by induction that if $q_\sigma(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\,x_0, (1 - \bar{\alpha}_t)I)$, then marginalizing out $x_t$ gives $q_\sigma(x_{t-1} \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\,x_0, (1 - \bar{\alpha}_{t-1})I)$ for **every** valid choice of $\sigma_t \in [0, \sqrt{1 - \bar{\alpha}_{t-1}}]$.

**(b)** Explain in 2 sentences why preserving the marginals $q(x_t \mid x_0)$ means the **exact same pretrained DDPM checkpoint** is already optimal for DDIM at $\eta = 0$ (deterministic) and on strided subsequences $\tau_1 < \tau_2 < \dots < \tau_S$.

### Problem (`a1_samplers`): Implement DDPM and Strided DDIM Samplers (7 points)

Implement `[adapters.run_ddpm_step]` and `[adapters.run_ddim_step]` (handling `t_prev_idx = -1` cleanly as $\bar{\alpha}_{-1} = 1.0$ and $\sigma_t = 0$) and pass all Assignment 1 unit tests:

```bash
pytest tests/test_a1_diffusion.py
```

### Problem (`a1_2d_experiment`): NFE Sweep and Latent Interpolation on 2D Checkerboard (8 points)

1. Build a 4-layer sinusoidal-time MLP (`d_hidden=128`, SiLU) in `cme296_assignments/` and train it for `5,000` steps (`batch_size=256`, cosine schedule `T=1000`, `v` or `eps` prediction) on `sample_2d_checkerboard`.
2. Sample `4,000` points at step budgets $\text{NFE} \in \{10, 25, 50, 200, 1000\}$, comparing strided DDPM ($\eta = 1$) against deterministic DDIM ($\eta = 0$). Measure the fraction of generated points landing on valid checkerboard squares across NFE budgets.
3. Sample two random endpoints $z^{(A)}, z^{(B)} \sim \mathcal{N}(0, I_2)$, spherically interpolate (`slerp`) between them across 9 values $\omega \in [0, 1]$, and decode via deterministic DDIM ($\eta = 0$, 50 steps) vs. stochastic DDPM ($\eta = 1$).

**Deliverable:** Scatter plots + NFE accuracy table + 3-sentence analysis of why `slerp` preserves $\mathcal{N}(0, I)$ norm better than linear `lerp` in high dimensions.
