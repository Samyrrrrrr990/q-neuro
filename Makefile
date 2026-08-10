.PHONY: sync test lint smoke-experiment-zero experiment-zero sample-efficiency generator-shift robustness-sweep neuro-task-suite active-evidence dynamics-suite analyses figures

sync:
	uv sync --extra dev

test:
	uv run pytest

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

analyses:
	uv run python research/analyses/analyze_generator_shift.py
	uv run python research/analyses/analyze_neuro_task_suite.py
	uv run python research/analyses/analyze_active_evidence.py

figures:
	uv run python research/figures/generate_experiment_zero.py
	uv run python research/figures/generate_generator_shift.py
	uv run python research/figures/generate_robustness_sweep.py
	uv run python research/figures/generate_neuro_task_suite.py
	uv run python research/figures/generate_active_evidence.py
