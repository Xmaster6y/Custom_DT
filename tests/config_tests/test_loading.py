"""Test loading."""

import torch
from model import DecisionTransformerConfig, DecisionTransformerModel

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


class TestBasicLoading:
    def test_default_config(self):
        conf = DecisionTransformerConfig()
        assert conf.hidden_size == 128

    def test_default_model(self):
        conf = DecisionTransformerConfig()
        model = DecisionTransformerModel(conf)
        assert model.config.hidden_size == 128
