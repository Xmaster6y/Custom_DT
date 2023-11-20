"""
Fixtures for testing utils.
"""

import os
import pathlib
import sys

import chess
import pytest
import torch

import src.utils.translate as translate
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.utils.dataset import OnePlayerChessDataset, TwoPlayersChessDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIRECTORY = pathlib.Path(__file__).parent.absolute()
DETECT_PLATFORM = "auto"


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
    yield DecisionTransformerModel(conf)


@pytest.fixture(scope="module")
def default_64x12_model():
    conf = DecisionTransformerConfig(
        state_dim=768,
        act_dim=4672,
    )
    yield DecisionTransformerModel(conf)


@pytest.fixture(scope="module")
def default_64_chess_dataset():
    generator = torch.Generator()
    generator.manual_seed(42)
    yield TwoPlayersChessDataset(
        file_name=f"{DIRECTORY}/assets/test_stockfish_10.jsonl",
        board_to_tensor=translate.board_to_64tensor,
        act_dim=4672,
        state_dim=64,
        window_size=10,
        generator=generator,
        return_ids=True,
    )


@pytest.fixture(scope="module")
def default_64x12_chess_dataset():
    generator = torch.Generator()
    generator.manual_seed(42)
    yield TwoPlayersChessDataset(
        file_name=f"{DIRECTORY}/assets/test_stockfish_10.jsonl",
        board_to_tensor=translate.board_to_64x12tensor,
        act_dim=4672,
        state_dim=768,
        window_size=10,
        generator=generator,
        return_ids=True,
    )


@pytest.fixture(scope="module")
def op_64_chess_dataset():
    generator = torch.Generator()
    generator.manual_seed(42)
    yield OnePlayerChessDataset(
        file_name=f"{DIRECTORY}/assets/test_stockfish_10.jsonl",
        board_to_tensor=translate.board_to_64tensor,
        act_dim=4672,
        state_dim=64,
        window_size=10,
        generator=generator,
        return_ids=True,
    )


@pytest.fixture(scope="module")
def op_64x12_chess_dataset():
    generator = torch.Generator()
    generator.manual_seed(42)
    yield OnePlayerChessDataset(
        file_name=f"{DIRECTORY}/assets/test_stockfish_10.jsonl",
        board_to_tensor=translate.board_to_64x12tensor,
        act_dim=4672,
        state_dim=768,
        window_size=10,
        generator=generator,
        return_ids=True,
    )


@pytest.fixture(scope="session")
def stockfish_engine():
    cwd = os.getcwd()
    sys.path.append(cwd)
    if DETECT_PLATFORM == "auto":
        if sys.platform in ["linux"]:
            platform = "linux"
        elif sys.platform in ["win32", "cygwin"]:
            platform = "windows"
        elif sys.platform in ["darwin"]:
            platform = "macos"
        else:
            raise ValueError(f"Unknown platform {sys.platform}")
    else:
        platform = DETECT_PLATFORM

    if platform in ["linux", "macos"]:
        exec_re = "stockfish*"
    elif platform == "windows":
        exec_re = "stockfish*.exe"
    else:
        raise ValueError(f"Unknown platform {platform}")

    stockfish_root = list(pathlib.Path(f"{cwd}/stockfish-source/stockfish/").glob(exec_re))[0]
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_root)
    yield engine
    engine.close()
