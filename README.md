# Stanford CME 296 (Lectures 1–5) — Custom From-Scratch Assignments (CS336 Style)

Stanford's **[CME 296: Diffusion & Large Vision Models](https://cme296.stanford.edu/syllabus/)** (taught by Afshine & Shervine Amidi) is a fast-paced mathematical tour of modern visual generative modeling — and officially has **no homework assignments** (exams only).

This repository fills that gap with **five rigorous, from-scratch assignments** modeled directly after **[Stanford CS336: Language Modeling from Scratch](https://github.com/stanford-cs336/assignment1-basics)**, covering **100% of CME 296 Lectures 1 through 5** (250 points total):

| Assignment | CME 296 Lecture | Core Derivations, Primitives Built From Scratch, & Experiments | Points | Graded By |
| :--- | :--- | :--- | :---: | :--- |
| **[Assignment 1](handouts/assignment1_diffusion.md)** | **Lec 1: Discrete-Time Diffusion** | Markov jump & VP variance proofs, resolution log-SNR shift, linear & cosine schedules, full ELBO & Bayes posterior, $(\epsilon, x_0, v)$ + Min-SNR equivalence, ancestral DDPM & strided non-Markovian **DDIM** ($\eta \in [0, 1]$), `slerp` interpolation | **50** | `tests/test_a1_diffusion.py` |
| **[Assignment 2](handouts/assignment2_score_sde.md)** | **Lec 2: Score Matching & SDEs** | Partition-function cancellation, Hyvärinen **ISM** + **Hutchinson Rademacher trace** (`create_graph=True`), Vincent **DSM**, low-density mode blindness proof, **Annealed Langevin Dynamics**, continuum **VP-SDE & VE-SDE**, reverse Euler-Maruyama, Fokker-Planck to **PF-ODE** (one-half factor) | **50** | `tests/test_a2_score_sde.py` |
| **[Assignment 3](handouts/assignment3_flow_matching.md)** | **Lec 3: Flow Matching & Rectified Flow** | NF/CNF Jacobian bottlenecks, $t = 0 \leftrightarrow 1$ convention bridge, **CFM** marginal-gradient proof, OT paths, equal-NFE **Euler vs. Heun** solvers, crossing-chords analysis + **Reflow** & trajectory straightness, $5 \times 5$ **Grand Unification Rosetta Stone** | **50** | `tests/test_a3_flow_matching.py` |
| **[Assignment 4](handouts/assignment4_latent_guidance.md)** | **Lec 4: Latent Space & Guidance** | **VAE** ELBO & closed-form Gaussian KL, Stable Diffusion's `0.18215` **`LatentScaler`** SNR mystery, from-scratch **CLIP** vs. **SigLIP** (distributed scaling & $-10$ bias init), **Classifier Guidance** vs. **CFG** + **Lin et al. Std-Rescaling** | **50** | `tests/test_a4_latent_guidance.py` |
| **[Assignment 5](handouts/assignment5_architectures.md)** | **Lec 5: Vision Architectures** | Skip variance scaling, sinusoidal time & **FiLM U-Net ResBlock**, `patchify` / `unpatchify`, **2D Axial RoPE**, **`adaLN-Zero` DiT** (step-0 identity & 2-step gradient unblocking), **MM-DiT** double-stream joint attention, **CS336-style `DiT-XL/2` FLOPs & memory accounting** | **50** | `tests/test_a5_architectures.py` |

---

## How This Repository Works (CS336 Design Contract)

```text
CME296-Custom-assignment/
├── cme296_assignments/       # <-- Empty! Build all your modules from scratch here
│   └── __init__.py
├── cme296_datasets/          # Provided zero-download 2D & procedural shape datasets (CPU/MPS friendly)
│   └── toy.py
├── handouts/                 # Full assignment specs (math derivations + coding + experiments)
│   ├── assignment1_diffusion.md
│   ├── assignment2_score_sde.md
│   ├── assignment3_flow_matching.md
│   ├── assignment4_latent_guidance.md
│   └── assignment5_architectures.md
├── tests/
│   ├── adapters.py           # <-- Wire your code into these 28 typed glue hooks
│   ├── test_a1_diffusion.py  # Do NOT edit test_*.py files
│   ├── test_a2_score_sde.py
│   ├── test_a3_flow_matching.py
│   ├── test_a4_latent_guidance.py
│   └── test_a5_architectures.py
├── writeup_template.md       # Structured template for all written derivations & plots
├── AGENTS.md                 # Socratic AI policy (prevents Cursor/Claude from spoiling answers)
├── make_submission.sh
└── pyproject.toml
```

1. **`cme296_assignments/*`:** Starts intentionally blank. Organize your from-scratch PyTorch code here however you like, following each handout's **"What you can use"** rules.
2. **`tests/adapters.py`:** Contains 28 typed adapter stubs that initially raise `NotImplementedError`. Once you build a component in `cme296_assignments/`, wire the corresponding `run_*` hook in `tests/adapters.py` with thin glue code only.
3. **`tests/test_a1_*.py` – `tests/test_a5_*.py`:** Immutable unit tests checking analytical edge cases, numerical stability, second-order gradients, and mathematical equivalences. **Do not edit the test files.**
4. **`writeup_template.md`:** Copy to `writeup.md` (or typeset as `writeup.pdf`) to record your proofs, brainstorming answers, FLOPs tables, and experiment figures.

---

## Quickstart

### 1. Environment Setup

Works with `uv` (CS336 style) or standard `pip` / `venv` on CPU, Apple Silicon (`mps`), or NVIDIA GPUs (`cuda`):

```bash
# Option A: using uv
uv sync
uv run pytest

# Option B: standard Python / pip
pip install -e .
pytest
```

Out of the box, running `pytest` will collect **28 test suites** across Assignments 1–5 and fail with `NotImplementedError` until you wire each adapter in `tests/adapters.py`.

### 2. Recommended Workflow (Per Lecture)

1. Watch the **[CME 296 lecture](https://cme296.stanford.edu/syllabus/)** alongside the **[CME 296 VIP Cheatsheet](https://github.com/afshinea/stanford-cme-296-diffusion-large-vision-models)**.
2. Open `handouts/assignment<k>_*.md` and solve the **Written / Brainstorming** problems on paper first — they are specifically designed to expose the subtle traps before you write code.
3. Implement the primitives from scratch in `cme296_assignments/`, connect `tests/adapters.py`, and run:

```bash
pytest tests/test_a1_diffusion.py
```

4. Run the **Low-Resource Experiment** at the end of the handout using `cme296_datasets` (each runs in under 15–60 seconds on a laptop).

---

## Credits & References

- Course syllabus and cheatsheet structure inspired by **[Stanford CME 296: Diffusion & Large Vision Models](https://cme296.stanford.edu/)** by Afshine Amidi & Shervine Amidi.
- Assignment architecture (`adapters.py`, from-scratch constraints, FLOPs accounting, and `AGENTS.md`) inspired by **[Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)**.
