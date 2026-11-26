# Assignment 4 (Lecture 4): Latent Space (VAEs), Contrastive Alignment (CLIP/SigLIP), and Guidance

**Course:** Stanford CME 296 — Diffusion & Large Vision Models (Custom CS336-Style Curriculum)  
**Total Points:** 50 points (17 pts Written/Brainstorming · 25 pts Code via `pytest` · 8 pts Experiments)

---

## 1. Assignment Overview

Lecture 4 solves the two bottlenecks that prevented early pixel-space diffusion models from scaling to modern text-to-image synthesis:

1. **Perceptual Latent Compression:** Compressing redundant pixel grids $(3, H, W)$ into compact, smooth VAE latent grids $(c, H/f, W/f)$ before running diffusion or flow matching.
2. **Multimodal Conditioning and Guidance:** Aligning image and text representations via contrastive pretraining (**CLIP** vs. **SigLIP**) and steering generation at test time via **Classifier Guidance** and **Classifier-Free Guidance (CFG)**.

### What you will implement

1. Closed-form diagonal Gaussian KL, reparameterization trick, and `0.18215`-style `LatentScaler` (`run_vae_kl_loss`, `run_vae_reparameterize`, `run_latent_scaler`)
2. Numerically stable **CLIP** symmetric InfoNCE loss and **SigLIP** pairwise sigmoid contrastive loss from scratch (`run_clip_loss`, `run_siglip_loss`)
3. Autograd **Classifier Guidance** and **Classifier-Free Guidance** with **Std-Rescaling** (`run_classifier_guidance_score`, `run_cfg_combine`)

### What you can use

Following CS336's from-scratch rule: in `run_clip_loss` and `run_siglip_loss`, you may **not** call `F.cross_entropy`, `F.log_softmax`, or `F.logsigmoid` — implement numerically stable log-sum-exp subtraction and `logaddexp` directly.

> **Low-Resource Tip:**  
> Section 5 trains a tiny Conv-VAE ($16 \times 16 \times 1 \to 4 \times 4 \times 4$ latents) + conditional Latent Flow Matching model on `sample_procedural_shapes` (or MNIST if `torchvision` is installed). Runs end-to-end in **under 60 seconds on CPU** and **under 15 seconds on Apple MPS / CUDA**.

---

## 2. Variational Autoencoders and Latent Diffusion Variance Scaling

### Problem (`a4_vae_elbo_and_scaling`): Closed-Form Gaussian KL and the `0.18215` Mystery (6 points)

**(a)** Starting from $\log p_\theta(x) = \log \int p_\theta(x, z)\,\mathrm{d}z$, derive the VAE ELBO:

$$
\log p_\theta(x) \ge \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right] - D_{\text{KL}}\left(q_\phi(z \mid x) \,\|\, \mathcal{N}(0, I)\right)
$$

Then integrate by hand to prove that for $q_\phi(z \mid x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ in $\mathbb{R}^d$:

$$
D_{\text{KL}}\left(\mathcal{N}(\mu, \text{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I)\right) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\right)
$$

**(b) Brainstorm (Why Stable Diffusion Multiplies Latents by `0.18215`):**  
In Latent Diffusion VAEs, the KL weight $\beta_{\text{KL}} \approx 10^{-6}$ is kept tiny so reconstructions stay razor-sharp. As a side effect, the empirical aggregate latent distribution $z \sim q_\phi(z \mid x)$ ends up with standard deviation $\hat{\sigma}_z \approx 5.49$ ($\text{Var}(z) \approx 30.1$) instead of $1.0$.

Suppose you forget to rescale $z$ to unit variance and run a standard VP-DDPM ($x_t = \sqrt{\bar{\alpha}_t}\,z + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$) or Flow Matching ($x_t = (1-t)\epsilon + t z$) directly on raw latents with $\text{Var}(z) = 30$:

1. Compute the actual Signal-to-Noise Ratio at $\bar{\alpha}_t = 0.5$ (intended to be $\text{SNR} = 1$) and at $t = T$ ($\bar{\alpha}_T = 10^{-3}$). Is $x_T$ actually $\mathcal{N}(0, I)$?
2. What failure artifact will you see when you start reverse sampling from $x_T \sim \mathcal{N}(0, I)$ at test time?

### Problem (`a4_vae_code`): Implement VAE KL, Reparameterization, and LatentScaler (7 points)

Implement `[adapters.run_vae_kl_loss]`, `[adapters.run_vae_reparameterize]`, and `[adapters.run_latent_scaler]`, then verify:

```bash
pytest tests/test_a4_latent_guidance.py -k "vae or latent_scaler"
```

---

## 3. Multimodal Contrastive Representations: CLIP vs. SigLIP

### Problem (`a4_clip_vs_siglip_brainstorm`): Softmax Bottlenecks and SigLIP Bias Initialization (5 points)

**(a)** CLIP computes row-wise and column-wise $B \times B$ softmaxes across a global batch $B = K \cdot b$ distributed across $K$ GPUs (local batch $b$ per GPU). Explain what collective communications are required to compute the exact CLIP denominator across $K$ GPUs, and explain why **SigLIP** removes the global normalization coupling:

$$
\mathcal{L}_{\text{SigLIP}} = \frac{1}{B}\sum_{i=1}^B \sum_{j=1}^B \log\left(1 + \exp\left(-z_{ij}(t\,x_i^\top y_j + b)\right)\right), \qquad z_{ij} \in \{+1, -1\}
$$

**(b) Brainstorm:** At step 0 of training, normalized random image and text embeddings are nearly orthogonal ($x_i^\top y_j \approx 0$). In a batch of $B = 32{,}768$, there are $B$ positive pairs ($z_{ii} = 1$) and $B(B - 1) \approx 1.07 \times 10^9$ negative pairs ($z_{ij} = -1$).  
Compute the ratio of negative-pair gradient magnitude to positive-pair gradient magnitude at step 0 if SigLIP sets initial bias $b = 0$, vs. Zhai et al.'s initialization $b = -10$.

### Problem (`a4_contrastive_code`): Implement Numerically Stable CLIP and SigLIP From Scratch (10 points)

Implement `[adapters.run_clip_loss]` (using max-shifted log-sum-exp, clamping $e^{\text{logit\_scale}} \le 100$) and `[adapters.run_siglip_loss]` (using `torch.logaddexp(0, -z * logits)`), then test:

```bash
pytest tests/test_a4_latent_guidance.py -k "clip or siglip"
```

---

## 4. Classifier Guidance vs. Classifier-Free Guidance (CFG)

### Problem (`a4_guidance_brainstorm`): Implicit Temperature, Adversarial Gradients, and Oversaturation (6 points)

**(a)** Use Bayes' rule $p(x_t \mid c) = \frac{p(c \mid x_t)\,p(x_t)}{p(c)}$ to show that $\nabla_{x_t}\log p(x_t) + \gamma\,\nabla_{x_t}\log p(c \mid x_t)$ is the exact score of the tilted distribution $\tilde{p}_\gamma(x_t \mid c) \propto p(x_t)\,p(c \mid x_t)^\gamma$ (explaining why $p(c)$ disappears). Then substitute $\nabla_{x_t}\log p(c \mid x_t) = \nabla_{x_t}\log p(x_t \mid c) - \nabla_{x_t}\log p(x_t)$ and $\nabla_{x_t}\log p \propto -\epsilon_\theta$ to derive Ho & Salimans' CFG formula:

$$
\hat{\epsilon}_{\text{CFG}}(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w\left(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right)
$$

What value of $w$ corresponds to standard unguided conditional sampling ($\gamma = 1$)?

**(b)** Give **two concrete reasons** why Classifier Guidance underperforms CFG in practice (think about noise-corrupted classifiers and how gradient ascent on a discriminative CNN in pixel space behaves like FGSM adversarial attacks).

**(c)** Suppose $\epsilon_\theta(x_t, c)$ and $\epsilon_\theta(x_t, \varnothing)$ both have unit variance and correlation $\rho = 0.85$. Compute $\text{Std}(\hat{\epsilon}_{\text{CFG}})$ at $w = 10.0$. Explain why this variance blowup causes **burnt/oversaturated colors** in generated images and how Lin et al.'s `rescale_phi` fixes it.

### Problem (`a4_guidance_code`): Implement Classifier Guidance and Rescaled CFG (8 points)

Implement `[adapters.run_classifier_guidance_score]` and `[adapters.run_cfg_combine]`, and pass the full Assignment 4 suite:

```bash
pytest tests/test_a4_latent_guidance.py
```

---

## 5. End-to-End Experiment: Latent Flow Matching and CFG Sweep

### Problem (`a4_latent_cfg_experiment`): Latent Scaling Ablation and CFG Pareto Frontier (8 points)

On `sample_procedural_shapes(n_samples=8192, image_size=16)` (4 classes: circle, square, cross, X):

1. Train a tiny Conv-VAE (`16x16x1 -> 4x4x2` latent, $\beta_{\text{KL}} = 10^{-4}$, `1,500` steps) and compute its empirical `scale_factor` via `run_latent_scaler`.
2. Train a class-conditional Latent Flow Matching model (`p_drop = 0.15` null-class token `c = 4`, `2,500` steps) **with** vs. **without** `LatentScaler`, verifying your hypothesis from Problem `a4_vae_elbo_and_scaling(b)`.
3. Sweep CFG scale $w \in \{0.0, 1.0, 2.0, 4.0, 8.0, 15.0\}$ (with `rescale_phi=0.0` vs. `0.7`), decode generated latents through your VAE decoder, and plot **(i) decoded image grids**, **(ii) latent L2 norm**, and **(iii) class-prototype accuracy vs. intra-class sample variance** (the classic CFG quality–diversity trade-off curve).
