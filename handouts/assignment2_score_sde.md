# Assignment 2 (Lecture 2): Score Matching, Langevin Dynamics, SDEs, & PF-ODE

**Course:** Stanford CME 296 — Diffusion & Large Vision Models (Custom CS336-Style Curriculum)  
**Total Points:** 50 points (18 pts Written/Brainstorming · 24 pts Code via `pytest` · 8 pts Experiments)

---

## 1. Assignment Overview

Lecture 2 reframes diffusion through **score functions** $s(x) = \nabla_x \log p(x)$ and **continuous-time Stochastic Differential Equations (SDEs)** — showing that discrete DDPM (Lecture 1) and Noise Conditional Score Networks (NCSN) are two discretizations of one continuous framework, and revealing the deterministic **Probability Flow ODE (PF-ODE)** hiding inside every SDE.

### What you will implement
1. Hyvärinen's **Implicit Score Matching (ISM)** with Hutchinson trace estimation & 2nd-order autograd (`run_ism_hutchinson_loss`)
2. Variance-balanced **Denoising Score Matching (DSM)** (`run_dsm_loss`)
3. **Annealed Langevin Dynamics (ALD)** across geometric noise scales (`run_annealed_langevin_dynamics`)
4. Continuous-time **VP-SDE** and **VE-SDE** drift, diffusion, and perturbation kernels (`run_vp_sde_params`, `run_ve_sde_params`)
5. **Reverse-time Euler-Maruyama SDE** step & **Probability Flow ODE** drift (`run_reverse_sde_step`, `run_pf_ode_drift`)

> **CME 296 Time Convention (Lecture 2):**  
> Continuous time $t \in [0, T]$ (with $T = 1$): clean data at $t = 0$, pure/maximal noise at $t = 1$.

> **Low-Resource Tip:**  
> Section 5 trains a tiny 3-layer MLP score network on `sample_2d_asymmetric_gmm` ($80\% / 20\%$ separated modes in $\mathbb{R}^2$). Both training and sampling complete in **< 15 seconds on CPU**.

---

## 2. Score Estimation: Explicit, Implicit (Hutchinson), & Denoising Score Matching

### Problem (`a2_ism_dsm_proof`): Partition Functions, Hyvärinen's Trick, & Vincent's Identity (6 points)
**(a)** For an energy-based model $p_\theta(x) = \frac{1}{Z_\theta}\exp(-E_\theta(x))$ where $Z_\theta = \int \exp(-E_\theta(u))\,\mathrm{d}u$, show why $\nabla_x \log p_\theta(x)$ is completely independent of $Z_\theta$ (whereas $\nabla_\theta \log p_\theta(x)$ is NOT!).  
**(b)** Prove Hyvärinen's identity (assuming $p(x)s_\theta(x) \to 0$ as $\|x\| \to \infty$) via integration by parts:
$$\mathbb{E}_{p(x)}\!\left[\frac{1}{2}\|s_\theta(x) - \nabla_x \log p(x)\|_2^2\right] = \mathbb{E}_{p(x)}\!\left[\operatorname{tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|_2^2\right] + C$$
Explain why computing $\operatorname{tr}(\nabla_x s_\theta(x))$ exactly via reverse-mode autodiff requires **$d$ separate backward passes** ($d = 196{,}608$ for a $256\times 256\times 3$ image!), and prove that for any probe $v$ with $\mathbb{E}[vv^\top] = I$, Hutchinson's estimator $\mathbb{E}_v[v^\top \nabla_x (v^\top s_\theta(x))]$ equals $\operatorname{tr}(\nabla_x s_\theta(x))$ in **1 vector-Jacobian product**! *(Bonus 1-liner: Why do Rademacher probes $v_i \in \{-1, +1\}$ have strictly lower variance on diagonal entries than Gaussian probes $v_i \sim \mathcal{N}(0, 1)$?)*  
**(c)** Prove Vincent's (2011) Denoising Score Matching equivalence: minimizing $\mathbb{E}_{p_\sigma(\tilde{x})}[\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}}\log p_\sigma(\tilde{x})\|_2^2]$ has the **exact same gradient wrt $\theta$** as minimizing $\mathbb{E}_{p(x)}\mathbb{E}_{q_\sigma(\tilde{x}\mid x)}[\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}}\log q_\sigma(\tilde{x}\mid x)\|_2^2]$.

### Problem (`a2_score_losses`): Implement Hutchinson ISM & Variance-Balanced DSM (8 points)
1. Implement `[adapters.run_ism_hutchinson_loss]` using `torch.autograd.grad(..., create_graph=True)` so second-order parameter gradients flow cleanly on `.backward()`.
2. Implement `[adapters.run_dsm_loss]` with weighting $\lambda(\sigma) = \sigma^2$. *(Think: without $\sigma^2$, what happens to $\mathbb{E}[\|\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}\mid x)\|_2^2] = d / \sigma^2$ as $\sigma \to 0.01$?)*

Run:
```bash
pytest tests/test_a2_score_sde.py -k "ism or dsm"
```

---

## 3. The Low-Density Blindspot & Annealed Langevin Dynamics

### Problem (`a2_low_density_brainstorm`): Why Unperturbed Score Matching Gets Mode Weights Wrong (4 points)
Consider an asymmetric mixture of two narrow Gaussians in $\mathbb{R}^d$ separated by $\|\mu_1 - \mu_2\| \gg \sigma_0$:
$$p(x) = \pi_1\,\mathcal{N}(x;\; \mu_1, \sigma_0^2 I) + \pi_2\,\mathcal{N}(x;\; \mu_2, \sigma_0^2 I), \qquad \pi_1 = 0.8,\; \pi_2 = 0.2$$
**(a)** Compute the exact analytical score $\nabla_x \log p(x)$ at any point $x$ within $3\sigma_0$ of $\mu_1$ (where $\mathcal{N}(x; \mu_2, \sigma_0^2 I) / \mathcal{N}(x; \mu_1, \sigma_0^2 I) \approx 0$). Notice what cancels out completely!  
**(b)** Suppose you initialize Langevin dynamics uniformly over a bounding box and run unannealed Langevin steps using the exact score $\nabla_x \log p(x)$. What fraction of samples end up in Mode 1 vs. Mode 2, and why did it completely miss $\pi_1 = 0.8$ vs. $\pi_2 = 0.2$? How does adding large noise $\sigma_1 \gg \|\mu_1 - \mu_2\|$ in **Annealed Langevin Dynamics** fix this?

### Problem (`a2_ald_sampler`): Implement Annealed Langevin Dynamics (5 points)
Implement `[adapters.run_annealed_langevin_dynamics]` and verify:
```bash
pytest tests/test_a2_score_sde.py -k langevin
```

---

## 4. SDE Unification, Reverse-Time Euler-Maruyama, & Probability Flow ODE

Forward SDE:
$$\mathrm{d}x = f(x, t)\,\mathrm{d}t + g(t)\,\mathrm{d}W$$
Anderson's (1982) Reverse-time SDE (integrating backwards in time):
$$\mathrm{d}x = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]\mathrm{d}t + g(t)\,\mathrm{d}\bar{W}$$
Song et al.'s (2021) Deterministic **Probability Flow ODE (PF-ODE)**:
$$\frac{\mathrm{d}x}{\mathrm{d}t} = f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

### Problem (`a2_sde_and_fokker_planck`): Continuum SDE Limits & The Factor of $\frac{1}{2}$ (8 points)
**(a)** Starting from discrete DDPM $x_i = \sqrt{1 - \beta_i}\,x_{i-1} + \sqrt{\beta_i}\,z_{i-1}$ with $\beta_i = \beta(t_i)\,\Delta t$ ($\Delta t = 1/N \to 0$), use first-order Taylor expansion $\sqrt{1 - \beta(t)\Delta t} \approx 1 - \frac{1}{2}\beta(t)\Delta t$ to derive the **VP-SDE** drift $f(x, t)$ and diffusion $g(t)$, and integrate the linear ODE for the mean to prove $p_{0t}(x(t) \mid x(0)) = \mathcal{N}\!\left(x(0)e^{-\frac{1}{2}\int_0^t \beta(s)\mathrm{d}s},\; \left(1 - e^{-\int_0^t \beta(s)\mathrm{d}s}\right)I\right)$.  
**(b)** The Fokker-Planck PDE governing the marginal density $p_t(x)$ of $\mathrm{d}x = f(x,t)\mathrm{d}t + g(t)\mathrm{d}W$ is:
$$\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot \big(f(x, t) p_t(x)\big) + \frac{1}{2}g(t)^2 \Delta p_t(x)$$
Substitute the identity $\Delta p_t(x) = \nabla \cdot (\nabla p_t(x)) = \nabla \cdot \big(p_t(x)\nabla \log p_t(x)\big)$ into Fokker-Planck to rewrite it in the pure continuity-equation form $\frac{\partial p_t}{\partial t} = -\nabla \cdot (\tilde{f}_t(x) p_t(x))$, directly deriving the **PF-ODE** drift $\tilde{f}_t(x)$ and explaining where the $\frac{1}{2}$ comes from!

### Problem (`a2_sde_code`): Implement VP/VE-SDEs, Reverse Euler-Maruyama, & PF-ODE (11 points)
Implement `[adapters.run_vp_sde_params]`, `[adapters.run_ve_sde_params]`, `[adapters.run_reverse_sde_step]`, and `[adapters.run_pf_ode_drift]`, then pass the full Assignment 2 test suite:
```bash
pytest tests/test_a2_score_sde.py
```

---

## 5. End-to-End Experiment: Proving Low-Density Blindness Empirically

### Problem (`a2_mode_proportions_experiment`): Single-Scale vs. Annealed / SDE / PF-ODE on `80/20` GMM (8 points)
Using `sample_2d_asymmetric_gmm(n_samples=8192, weight_mode1=0.8, sigma=0.25, separation=5.0)`:
1. Train a single-noise score MLP at $\sigma = 0.25$ via DSM (`3,000` steps) and sample `2,000` points initialized uniformly in $[-4, 4]^2$ via fixed-$\sigma$ Langevin dynamics. Record the empirical fraction of samples in Mode 1 ($x_1 + x_2 < 0$).
2. Train a noise-conditional score MLP across $L = 10$ geometric scales $\sigma_1 = 6.0 \to \sigma_{10} = 0.20$ (or continuous VE/VP-SDE) and sample `2,000` points via **(i) Annealed Langevin Dynamics**, **(ii) Reverse-time SDE**, and **(iii) PF-ODE**.
3. Compare recovered Mode-1 mass against the ground-truth `0.80`!  
*Deliverable:* Mode-weight table across all 4 samplers + vector-field quiver plot at $\sigma = 4.0$ vs. $\sigma = 0.25$ + 3-sentence reflection connecting your empirical numbers back to Problem `a2_low_density_brainstorm`.
