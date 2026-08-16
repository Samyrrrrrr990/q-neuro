.PHONY: sync test lint smoke-experiment-zero experiment-zero sample-efficiency generator-shift robustness-sweep neuro-task-suite active-evidence dynamics-suite ablation-suite observable-probe training-laws hard-halting trajectories discovery simulator-red-team shift-gauntlet shift-pilot smoke-shift-pilot mechanism-suite smoke-mechanism-suite computational-law-suite independent-task-audit independent-discovery smoke-independent-discovery freeze-candidate-law independent-confirmation smoke-independent-confirmation qn-grand-001 qe-000001 qe-000002 qe-000003 qe-000004 qe-000006 qe-000008 qe-000009 qe-000010 qe-all discovery-suite claim-audit dashboard analyses figures paper-tables paper-source latex paper paper-release verify-release reproduce-paper

sync:
	uv sync --extra dev

test:
	uv run python -m pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

smoke-experiment-zero:
	uv run python -m experiments.run_experiment_zero \
		--config experiments/configs/experiment_zero.yaml --smoke

experiment-zero:
	uv run python -m experiments.run_experiment_zero \
		--config experiments/configs/experiment_zero.yaml

sample-efficiency:
	uv run python -m experiments.run_sample_efficiency \
		--config experiments/configs/experiment_zero_sample_efficiency.yaml

generator-shift:
	uv run python -m experiments.run_generator_shift \
		--config experiments/configs/experiment_zero_generator_shift.yaml

robustness-sweep:
	uv run python -m experiments.run_robustness_sweep \
		--config experiments/configs/robustness_world_sweep.yaml

neuro-task-suite:
	uv run python -m experiments.run_neuro_task_suite \
		--config experiments/configs/neuro_task_suite.yaml

active-evidence:
	uv run python -m experiments.run_active_evidence \
		--config experiments/configs/active_evidence.yaml

dynamics-suite:
	uv run python -m experiments.run_dynamics_suite \
		--config experiments/configs/dynamics_suite.yaml

ablation-suite:
	uv run python -m experiments.run_dynamics_suite \
		--config experiments/configs/ablation_suite.yaml

observable-probe:
	uv run python -m experiments.run_observable_probe \
		--config experiments/configs/observable_probe.yaml

training-laws:
	uv run python -m experiments.run_training_laws \
		--config experiments/configs/training_laws.yaml

hard-halting:
	uv run python -m experiments.run_hard_halting \
		--config experiments/configs/hard_halting.yaml

trajectories:
	uv run python -m experiments.run_trajectory_study \
		--config experiments/configs/trajectory_study.yaml

discovery:
	uv run python -m experiments.run_discovery_engine \
		--config experiments/configs/discovery_engine.yaml

simulator-red-team:
	uv run python -m experiments.run_simulator_red_team \
		--config experiments/configs/simulator_red_team_v2.yaml

shift-gauntlet:
	uv run python -m experiments.run_shift_gauntlet \
		--config experiments/configs/shift_gauntlet.yaml

smoke-shift-pilot:
	uv run python -m experiments.run_shift_pilot \
		--config experiments/configs/shift_pilot.yaml --smoke

shift-pilot:
	uv run python -m experiments.run_shift_pilot \
		--config experiments/configs/shift_pilot.yaml

smoke-mechanism-suite:
	uv run python -m experiments.run_mechanism_suite \
		--config experiments/configs/mechanism_suite.yaml --smoke

mechanism-suite:
	uv run python -m experiments.run_mechanism_suite \
		--config experiments/configs/mechanism_suite.yaml

computational-law-suite:
	uv run python -m experiments.run_computational_law_suite \
		--config experiments/configs/computational_law_suite.yaml

independent-task-audit:
	uv run python -m experiments.run_independent_task_audit \
		--config experiments/configs/independent_task_audit.yaml

smoke-independent-discovery:
	uv run python -m experiments.run_independent_discovery \
		--config experiments/configs/independent_discovery.yaml --smoke

independent-discovery:
	uv run python -m experiments.run_independent_discovery \
		--config experiments/configs/independent_discovery.yaml

freeze-candidate-law:
	uv run python -m research.freeze_candidate_law

smoke-independent-confirmation:
	uv run python -m experiments.run_independent_confirmation \
		--config experiments/configs/independent_confirmation.yaml --smoke

independent-confirmation:
	uv run python -m experiments.run_independent_confirmation \
		--config experiments/configs/independent_confirmation.yaml

qn-grand-001:
	uv run python -m experiments.run_qn_grand_001 \
		--config experiments/configs/qn_grand_001.yaml

claim-audit:
	uv run python -m research.adversarial_reviewer

dashboard:
	uv run python scripts/build_dashboard_data.py

analyses:
	uv run python -m research.analyses.analyze_generator_shift
	uv run python -m research.analyses.analyze_neuro_task_suite
	uv run python -m research.analyses.analyze_active_evidence
	uv run python -m research.analyses.analyze_dynamics_suite
	uv run python -m research.analyses.analyze_ablation_suite
	uv run python -m research.analyses.analyze_observable_probe
	uv run python -m research.analyses.analyze_training_laws
	uv run python -m research.analyses.analyze_hard_halting
	uv run python -m research.analyses.analyze_trajectories
	uv run python -m research.analyses.analyze_falsification_phase

figures:
	uv run python -m research.figures.generate_experiment_zero
	uv run python -m research.figures.generate_generator_shift
	uv run python -m research.figures.generate_robustness_sweep
	uv run python -m research.figures.generate_neuro_task_suite
	uv run python -m research.figures.generate_active_evidence
	uv run python -m research.figures.generate_dynamics_suite
	uv run python -m research.figures.generate_ablation_suite
	uv run python -m research.figures.generate_observable_probe
	uv run python -m research.figures.generate_training_laws
	uv run python -m research.figures.generate_hard_halting
	uv run python -m research.figures.generate_trajectory_signature
	uv run python -m research.figures.generate_paper_extended
	uv run python -m research.figures.generate_falsification_phase

paper-tables:
	uv run python paper/build_tables.py

paper-source: paper-tables
	uv run --extra paper python paper/build_manuscript.py

latex: paper-source
	@if command -v latexmk >/dev/null 2>&1; then \
		(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && mv main.pdf qneuro.pdf); \
	else \
		echo "latexmk is not installed; modular LaTeX source was generated and DOCX rendering supplies qneuro.pdf."; \
	fi

paper: figures paper-source
	@command -v soffice >/dev/null 2>&1 || { echo "LibreOffice (soffice) is required to render qneuro.pdf"; exit 1; }
	@mkdir -p paper/rendered
	soffice --headless --convert-to pdf --outdir paper/rendered paper/qneuro.docx

paper-release: figures paper-source
	@command -v soffice >/dev/null 2>&1 || { echo "LibreOffice (soffice) is required to render qneuro.pdf"; exit 1; }
	soffice --headless --convert-to pdf --outdir paper paper/qneuro.docx

qe-000001:
	uv run python -m experiments.run_qe_000001

qe-000002:
	uv run python -m experiments.run_qe_000002

qe-000003:
	uv run python -m experiments.run_qe_000003

qe-000004:
	uv run python -m experiments.run_qe_000004

qe-000006:
	uv run python -m experiments.run_qe_000006

qe-000008:
	uv run python -m experiments.run_qe_000008

qe-000009:
	uv run python -m experiments.run_qe_000009

qe-000010:
	uv run python -m experiments.run_qe_000010

discovery-suite:
	uv run python -m research.discovery_lab.run_discovery_001

qe-all: qe-000001 qe-000002 qe-000003 qe-000004 qe-000006 qe-000008 qe-000009 qe-000010
	@echo "Ran the registered equivalence experiments; see experiments/results/QE-*."

verify-release:
	uv run python scripts/verify_release.py

reproduce-paper: test lint dashboard figures paper verify-release
	@echo "Rebuilt tests, dashboard data, figures, tables, LaTeX, DOCX, and an ignored verification PDF from locked sources."
