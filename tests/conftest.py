"""
Fixtures for testing utils.
"""

import pytest

from src.metric.stockfish import StockfishMetric

DETECT_PLATFORM = "auto"


@pytest.fixture(scope="session")
def stockfish_metric():
    try:
        metric = StockfishMetric(default_platform=DETECT_PLATFORM)
        yield metric
    finally:
        metric.engine.quit()
