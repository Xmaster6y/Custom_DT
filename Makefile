# CI
.PHONY: checks
checks:
	poetry run pre-commit run --all-files

.PHONY: tests
tests:
	poetry run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=40 -s

# Training
.PHONY: train
train:
	python -m src.train.window_autoreg

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir logging
