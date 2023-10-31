# CI
.PHONY: checks
checks:
	pre-commit run --all-files

.PHONY: tests
tests:
	pytest tests --cov=src --cov-report=term-missing --cov-fail-under=1 -s

# Training
.PHONY: train
train:
	python -m src.train.window_autoreg

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir logging
