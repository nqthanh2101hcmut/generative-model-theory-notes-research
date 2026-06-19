# Detailed Technical Notes

This folder contains long-form academic notes for the research direction:

> Optimization, statistical and variational foundations of diffusion and flow-based generative models.

## Markdown reports

- `00_research_map_and_learning_roadmap.md` - research map, mathematical layers and 12-month roadmap.
- `01_optimization_variational_analysis.md` - KKT, KL regularization, primal-dual methods and variational convergence.
- `02_probability_statistics_foundations.md` - probability measures, empirical distributions, weak/Wasserstein convergence and error decomposition.
- `03_optimal_transport_entropy_schrodinger.md` - Kantorovich OT, entropic OT, Sinkhorn, OT-CFM and Schrödinger bridge.
- `04_diffusion_score_based_models.md` - diffusion SDEs, Fokker-Planck, score matching, reverse SDE and sampling error.
- `05_flow_matching_continuous_flows.md` - ODE flows, continuity equation, conditional flow matching and OT paths.
- `06_research_program_and_paper_plan.md` - PhD thesis structure, paper plan, theorem roadmap and experiments.
- `ALL_NOTES_MASTER_REPORT.md` - all reports combined into one master document.

## PDF versions

Compiled PDFs are placed in:

```text
notes/pdf/
```

Use the PDFs for reading and the Markdown files for editing/version control.

## How to rebuild PDFs

From the repository root:

```bash
make pdf-notes
```

or run Pandoc manually:

```bash
pandoc notes/technical-reports/ALL_NOTES_MASTER_REPORT.md \
  --pdf-engine=xelatex \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  -V geometry:margin=2.4cm \
  -o notes/pdf/ALL_NOTES_MASTER_REPORT.pdf
```
