# Assignment 3 (Lecture 3): Flow Matching, ODE Solvers, Rectified Flow, and Grand Unification

**Course:** Stanford CME 296 — Diffusion & Large Vision Models (Custom CS336-Style Curriculum)  
**Total Points:** 50 points (18 pts Written/Brainstorming · 24 pts Code via `pytest` · 8 pts Experiments)

---

## 1. Assignment Overview

Lecture 3 traces the evolution from classical **Normalizing Flows** and **Continuous Normalizing Flows (CNFs)** to simulation-free **Conditional Flow Matching (CFM)** and **Rectified Flow (Reflow)** — then unifies Diffusion, Score Matching, and Flow Matching as coordinate reparameterizations of a single Gaussian probability path.

### What you will implement

1. Time and sign convention bridge between CME 296 Flow Matching and Diffusion (`run_convert_time_convention`)
2. Optimal-Transport linear conditional path and CFM regression loss (`run_ot_conditional_path`, `run_cfm_loss`)
3. 1st-order Euler and 2nd-order Heun ODE solvers (`run_ode_euler_step`, `run_ode_heun_step`, `run_flow_sample_loop`)
4. **Rectified Flow** path straightness metric and **Reflow** coupling (`run_trajectory_straightness`)
5. **Grand Unification Rosetta Stone** ($5 \times 5$ conversions across $x_{\text{data}}, \epsilon, s, v, u_{\text{FM}}$ via `run_rosetta_conversions`)

> **CRITICAL CME 296 Convention Warning (Lecture 3 Cheatsheet Section 1.3):**  
> In CME 296 Lecture 3, **noisy prior is** $t = 0$ ($x_0 \sim \mathcal{N}(0, I)$) and **clean data is** $t = 1$ ($x_1 \sim p_{\text{data}}$) — the **exact opposite** of Lectures 1–2 (and opposite of papers like *Self-Flow*, which put clean data at $t = 0$ and noise at $t = 1$). Mastering this sign/time flip in Problem `a3_convention_bridge` will save you countless debugging hours in research codebases.

---

## 2. Normalizing Flows, CNFs, and Time Conventions

### Problem (`a3_cnf_bottleneck`): Why Classical Flows and Likelihood CNFs Did Not Scale (4 points)

**(a)** For an invertible bijection $x_1 = f_\theta(x_0)$ with $x_0 \sim p_0$, state the change-of-variables log-density formula for $\log p_1(x_1)$. What is the time complexity in dimension $d$ of computing an unrestricted $\log |\det J_{f_\theta}(x_0)|$, and what architectural compromises did RealNVP / Glow have to make to avoid it?

**(b)** Continuous Normalizing Flows replace $\log |\det J|$ with Chen et al.'s instantaneous trace identity $\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(\psi_t(x_0)) = -\nabla \cdot u_\theta(\psi_t(x_0), t)$, freeing the architecture — yet maximum-likelihood CNF training was still 100x too slow for ImageNet. Why? Name the two expensive operations inside every single training step.

### Problem (`a3_convention_bridge`): Mastering the $t = 0 \leftrightarrow 1$ Sign Flip (4 points: 2 Written + 2 Code)

**(a) Written:** Suppose Model A uses the CME 296 Lecture 3 convention ($x_t = (1-t) x_{\text{noise}} + t x_{\text{data}}$, velocity $u_t = \mathrm{d}x / \mathrm{d}t$) and Model B uses the *Self-Flow* / Diffusion convention ($x_\tau = (1-\tau) x_{\text{data}} + \tau x_{\text{noise}}$, velocity $v_\tau = \mathrm{d}x / \mathrm{d}\tau$). Write $\tau$ in terms of $t$ and prove via the chain rule how $v_\tau(x)$ relates to $u_t(x)$ at the same state $x$.

**(b) Code:** Implement `[adapters.run_convert_time_convention]` and run:

```bash
pytest tests/test_a3_flow_matching.py -k test_time_convention_bridge
```

---

## 3. Conditional Flow Matching (CFM) and ODE Solvers

### Problem (`a3_cfm_gradient_proof`): Why Regressing Random Chords Recovers the Marginal Flow (5 points)

The intractable **Marginal Flow Matching** objective is $\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t,\, p_t(x)}\left[\|u_\theta(x, t) - u_t(x)\|_2^2\right]$, where the marginal vector field generating $p_t(x) = \int p_t(x \mid z)\,q(z)\,\mathrm{d}z$ is:

$$
u_t(x) = \int u_t(x \mid z)\,\frac{p_t(x \mid z)\,q(z)}{p_t(x)}\,\mathrm{d}z = \mathbb{E}_{q(z \mid x_t = x)}\left[u_t(x \mid z)\right]
$$

Prove **without using PDEs** (expanding the squared norm and applying the law of total expectation) that:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, z \sim q(z),\, x \sim p_t(x \mid z)}\left[\|u_\theta(x, t) - u_t(x \mid z)\|_2^2\right] = \mathcal{L}_{\text{FM}}(\theta) + C
$$

where $C$ is independent of $\theta$, so $\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta) \equiv \nabla_\theta \mathcal{L}_{\text{FM}}(\theta)$.

### Problem (`a3_cfm_and_solvers`): Implement OT Path, CFM Loss, and Euler / Heun Solvers (10 points)

1. Implement `[adapters.run_ot_conditional_path]` and `[adapters.run_cfm_loss]`.
2. Implement `[adapters.run_ode_euler_step]`, `[adapters.run_ode_heun_step]`, and `[adapters.run_flow_sample_loop]`.  
   *(Note:* When comparing Euler vs. Heun fairly, always compare at **equal NFE** — `S` Heun steps make `2S` network evaluations.*)*

Run:

```bash
pytest tests/test_a3_flow_matching.py -k "cfm or euler_vs_heun"
```

---

## 4. Rectified Flow and Reflow: Why Straight Chords Learn Curved ODEs

### Problem (`a3_rectified_flow_brainstorm`): Crossing Chords and Why Reflow Straightens Paths (5 points)

**(a) Brainstorm:** During CFM / 1-Rectified Flow training, every single conditional path $x_t = (1-t) x_0 + t x_1$ is a **perfect straight line** with constant velocity $x_1 - x_0$. Why, then, are the trajectories generated by integrating $\mathrm{d}z_t / \mathrm{d}t = u_\theta(z_t, t)$ at test time **curved** instead of straight?  
**Hint:** Pick $x_0^{(a)} = (-1, 0),\; x_1^{(a)} = (1, 0)$ and $x_0^{(b)} = (0, -1),\; x_1^{(b)} = (0, 1)$. Where do their straight chords intersect at $t = 0.5$, what must a deterministic vector field $u_{0.5}(0, 0)$ output there, and can two solutions of a smooth ODE ever cross in $(x, t)$ space?

**(b)** Explain how **Reflow** — generating coupled pairs $(Z_0, \hat{Z}_1 = \text{ODESolve}(Z_0, u_\theta^{(1)}))$ and retraining a 2-Rectified Flow $u_\theta^{(2)}$ on linear interpolations of $(Z_0, \hat{Z}_1)$ — eliminates chord crossings, and explain why zero straightness error $S(Z) = 0$ implies **1-step Euler** $z_1 = z_0 + u_\theta(z_0, 0)$ **is exact**.

### Problem (`a3_straightness_code`): Implement Trajectory Straightness Metric (4 points)

Implement `[adapters.run_trajectory_straightness]` and test:

```bash
pytest tests/test_a3_flow_matching.py -k straightness
```

---

## 5. Grand Unification Rosetta Stone (Lectures 1, 2, and 3)

### Problem (`a3_rosetta_stone`): $5 \times 5$ Conversion Across General Gaussian Paths (10 points: 2 Written + 8 Code)

For **any** general Gaussian interpolant $x_t = \alpha_t x_{\text{data}} + \sigma_t \epsilon$ with $\alpha_t > 0,\; \sigma_t > 0$ (note: in VP-DDPM $\alpha_t^2 + \sigma_t^2 = 1$, whereas in OT Flow Matching $\alpha_t = t,\; \sigma_t = 1 - t$ so $\alpha_t^2 + \sigma_t^2 \neq 1$):

**(a) Written:** Write the $2 \times 2$ linear system expressing $(x_t, v)$ in terms of $(x_{\text{data}}, \epsilon)$ where $v = \alpha_t \epsilon - \sigma_t x_{\text{data}}$, invert the $2 \times 2$ matrix analytically without assuming $\alpha_t^2 + \sigma_t^2 = 1$, and do the same for $(x_t, u_{\text{FM}})$ where $u_{\text{FM}} = x_{\text{data}} - \epsilon$.

**(b) Code:** Implement `[adapters.run_rosetta_conversions]` and pass the entire Assignment 3 test suite:

```bash
pytest tests/test_a3_flow_matching.py
```

### Problem (`a3_reflow_2d_experiment`): 1-Rectified Flow vs. 2-Rectified Flow on Two-Moons (8 points)

1. Train a 4-layer MLP velocity network $u_\theta^{(1)}$ (`4,000` steps, `batch_size=256`) mapping $x_0 \sim \mathcal{N}(0, I_2)$ to `sample_2d_two_moons` ($x_1$).
2. Generate `16,384` coupled Reflow pairs $(Z_0, \hat{Z}_1)$ by integrating $u_\theta^{(1)}$ with 50 Heun steps (`NFE = 100`), then train $u_\theta^{(2)}$ (`4,000` steps) on those coupled pairs.
3. Compare **(i) `trajectory_straightness`**, **(ii) 20-particle trajectory plots**, and **(iii) 1-step and 2-step Euler sample quality** between $u_\theta^{(1)}$ (1-RF) and $u_\theta^{(2)}$ (2-RF).

**Deliverable:** Side-by-side trajectory plots (showing curved vs. straightened paths), straightness numbers, and 1-step Euler scatter comparison.
