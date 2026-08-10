.PHONY: sync test lint smoke-experiment-zero experiment-zero sample-efficiency

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
