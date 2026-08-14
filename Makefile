.PHONY: sync test lint smoke-experiment-zero experiment-zero sample-efficiency generator-shift robustness-sweep neuro-task-suite active-evidence dynamics-suite ablation-suite observable-probe training-laws hard-halting trajectories discovery dashboard analyses figures

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
