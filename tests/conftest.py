"""
Fixtures for testing utils.
"""

import pytest

from src.metric.stockfish import StockfishMetric

DETECT_PLATFORM = "auto"


@pytest.fixture(scope="function")
def stockfish_metric():
    try:
        metric = StockfishMetric(default_platform=DETECT_PLATFORM)
        yield metric
    finally:
        metric.engine.quit()


@pytest.fixture(scope="function")
def stockfish_metric_fix_compose():
    try:
        metric = StockfishMetric(default_platform=DETECT_PLATFORM)
        yield metric
    finally:
        metric.engine.quit()
