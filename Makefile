#CI
.PHONY: checks
checks:
	pre-commit run --all-files

.PHONY: tests
tests:
	pytest tests --cov=src --cov-report=term-missing --cov-fail-under=1 -s
