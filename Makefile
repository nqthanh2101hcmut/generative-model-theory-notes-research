.PHONY: run-basic run-ot run-flow run-score

run-basic:
	python projects/01_simplex_kl_optimization/simplex_kl.py
	python projects/02_primal_dual_constraints/primal_dual_simplex.py

run-ot:
	python projects/03_sinkhorn_ot/sinkhorn_demo.py

run-flow:
	python projects/04_flow_matching_2d/flow_matching_toy.py

run-score:
	python projects/05_score_matching_2d/score_matching_toy.py

PDF_ENGINE=xelatex
PANDOC_VARS=-V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -V geometry:margin=2.4cm -V papersize:a4 -V fontsize=11pt

pdf-notes:
	mkdir -p notes/pdf
	for f in notes/technical-reports/*.md; do \
	  b=$$(basename $$f .md); \
	  pandoc -f markdown+tex_math_single_backslash+tex_math_dollars "$$f" --pdf-engine=$(PDF_ENGINE) $(PANDOC_VARS) -o "notes/pdf/$$b.pdf"; \
	done

