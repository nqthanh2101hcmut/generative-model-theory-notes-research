# Generative Model Theory Notes

A structured self-study and research repository for the mathematical foundations of modern generative models, especially diffusion models, flow matching, optimal transport, and entropy-regularized distributional optimization.

This repository is designed for a PhD-oriented path in **Applied Mathematics**, with a long-term research profile in **Optimization, Probability, Statistics, and Theoretical Foundations of Generative Models**.

## Research direction

**Main theme**

> Optimization, statistical and variational foundations of diffusion and flow-based generative models.

**Core mathematical areas**

- Optimization theory: convex optimization, KKT, duality, primal-dual methods, mirror descent, dual averaging.
- Variational analysis: lower semicontinuity, coercivity, Gamma/epi convergence, stability of minimizers.
- Probability and statistics: probability measures, empirical measures, weak convergence, Wasserstein convergence, generalization error.
- Optimal transport: Kantorovich OT, Wasserstein distance, entropic OT, Sinkhorn algorithm, dynamic OT.
- Generative models: score matching, diffusion SDEs, flow matching, Schrödinger bridge, diffusion alignment.
- Numerical computing: ODE/SDE solvers, Euler-Maruyama, numerical stability, toy simulations.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── docs/
│   ├── 00_roadmap.md
│   ├── 01_research_agenda.md
│   ├── 02_literature_map.md
│   ├── 03_paper_reading_template.md
│   ├── 04_mathematical_symbols.md
│   └── 05_research_problem_bank.md
├── notes/
│   ├── optimization/
│   ├── probability-statistics/
│   ├── variational-analysis/
│   ├── optimal-transport/
│   ├── diffusion-flow/
│   └── numerical-computing/
├── papers/
│   ├── reading-list.md
│   ├── summary-template.md
│   └── summaries/
├── projects/
│   ├── 01_simplex_kl_optimization/
│   ├── 02_primal_dual_constraints/
│   ├── 03_sinkhorn_ot/
│   ├── 04_flow_matching_2d/
│   ├── 05_score_matching_2d/
│   └── 06_error_decomposition/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
└── templates/
```

## How to use this repository

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run toy projects

```bash
python projects/01_simplex_kl_optimization/simplex_kl.py
python projects/02_primal_dual_constraints/primal_dual_simplex.py
python projects/03_sinkhorn_ot/sinkhorn_demo.py
python projects/04_flow_matching_2d/flow_matching_toy.py
python projects/05_score_matching_2d/score_matching_toy.py
```

Figures will be saved to each project's `outputs/` folder.

### 3. Weekly workflow

Each week, produce three outputs:

1. One mathematical note in `notes/`.
2. One paper summary in `papers/summaries/`.
3. One code experiment in `projects/`.

Recommended weekly cycle:

```text
Monday-Tuesday: learn theory
Wednesday: solve/prove exercises
Thursday-Friday: implement code
Saturday: read one paper
Sunday: write summary and research log
```

## Target research profile

Use this repository to build a profile around:

> Variational analysis and optimization on probability spaces; optimality conditions, duality and stability of entropy-regularized optimization problems; probability, statistics and optimal transport methods for diffusion and flow-based generative models.

## Suggested first milestone

After 4 weeks, you should have:

- A note on KL-regularized optimization.
- A note on empirical measures and population vs empirical objectives.
- A working simplex KL optimization script.
- A working Sinkhorn OT script.
- One paper summary on flow matching or diffusion SDEs.

## License

MIT License. See `LICENSE`.

## Detailed technical reports

The `notes/technical-reports/` folder contains long-form academic notes written in Markdown, and `notes/pdf/` contains PDF versions for reading. The combined report is:

- Markdown: `notes/technical-reports/ALL_NOTES_MASTER_REPORT.md`
- PDF: `notes/pdf/ALL_NOTES_MASTER_REPORT.pdf`

These reports cover optimization, variational analysis, probability/statistics, optimal transport, diffusion models, flow matching, and a PhD-level research program.


## English Technical Report PDF

A revised English technical-report version of the research notes is available in:

- `notes/technical-reports-en/ALL_NOTES_MASTER_REPORT_EN.md`
- `notes/pdf/ALL_NOTES_MASTER_REPORT_EN.pdf`

The English reports are written in an academic research-note style and cover optimization, variational analysis, probability/statistics, optimal transport, diffusion models, flow matching, and a PhD-oriented paper plan.
