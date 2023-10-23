"""
Fixtures for testing utils.
"""

import pytest
import torch

from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(DEVICE)


@pytest.fixture(scope="module")
def simple_seq():
    return "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"


@pytest.fixture(scope="module")
def default_64_model():
    conf = DecisionTransformerConfig(
        state_dim=64,
        act_dim=4672,
    )
    model = DecisionTransformerModel(conf)
    yield model


@pytest.fixture(scope="module")
def default_64x12_model():
    conf = DecisionTransformerConfig(
        state_dim=768,
        act_dim=4672,
    )
    model = DecisionTransformerModel(conf)
    yield model
