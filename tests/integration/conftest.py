"""
Fixtures for testing utils.
"""

import os
import pathlib

import chess
import jsonlines
import pytest
import torch

import src.utils.translate as translate
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.train_rl.utils.rl_trainer_config import RLTrainerConfig
from src.utils.dataset import OnePlayerChessDataset, TwoPlayersChessDataset
from src.utils.leela_constants import ACT_DIM, STATE_DIM

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


@pytest.fixture(scope="module")
def default_dataset_boards():
    file_name = f"{DIRECTORY}/assets/test_stockfish_10.jsonl"
    games = []
    with jsonlines.open(file_name) as reader:
        games.extend(iter(reader))
    board_list = []
    for game in games:
        board = chess.Board()
        for alg_move in game["moves"].split():
            if alg_move.endswith("."):
                continue
            board.push_san(alg_move)
        board_list.append(board)
    yield board_list


@pytest.fixture(scope="module")
def RL_deprec_encoding_model():
    conf = DecisionTransformerConfig(
        state_dim=72,
        act_dim=4672,
        n_layers=6,
        n_heads=4,
        hidden_size=64 * 4,
    )
    yield DecisionTransformerModel(conf)


@pytest.fixture(scope="module")
def RL_trainer_cfg():
    cwd = os.getcwd()
    yield RLTrainerConfig(
        output_dir=f"{cwd}\\weights\\test_training",
        logging_dir=f"{cwd}\\logging\\test_training",
        figures_dir=f"{cwd}\\figures\\test_training",
        overwrite_output_dir=True,
        logging_steps_ratio=0.2,
        eval_steps_ratio=0.1,
        save_steps_ratio=0.2,
        num_train_epochs=10,
        run_name="test_training",
        seed=42,
        lr=1e-4,
        one_player=True,
        use_stock_fish_eval=True,
        stockfish_metric=None,
        stockfish_eval_depth=6,
        stockfish_gameplay_depth=2,
        resume_from_checkpoint=False,
        checkpoint_path=None,
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        temperature=1.0,
        lr_scheduler=False,
    )


@pytest.fixture(scope="module")
def RL_leela_encoding_model():
    conf = DecisionTransformerConfig(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        n_layers=6,
        n_heads=4,
        hidden_size=64 * 4,
    )
    yield DecisionTransformerModel(conf)
