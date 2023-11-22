"""
Dataset class for loading data from jsonl files.
"""

from typing import Callable, Optional

import chess
import jsonlines
import torch
from torch.utils.data import Dataset

import src.utils.translate as translate
from src.metric.stockfish import StockfishMetric


class TwoPlayersChessDataset(Dataset):
    def __init__(
        self,
        file_name: str,
        board_to_tensor: Optional[Callable[[chess.Board], torch.Tensor]],
        act_dim: int,
        state_dim: int,
        window_size: int,
        generator: torch.Generator,
        return_ids: bool = False,
        eval_mode: bool = False,
        shaping_rewards: bool = False,
        stockfish_metric: Optional[StockfishMetric] = None,
    ):
        self.games = []  # Can be heavy on memory, but good enough for now
        with jsonlines.open(file_name) as reader:
            self.games.extend(iter(reader))
        self.board_to_tensor = board_to_tensor
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.device = torch.device("cpu")  # Load full dataset on cpu
        self.window_size = window_size
        self.generator = generator
        self.return_ids = return_ids
        self.eval_mode = eval_mode
        self.shaping_rewards = shaping_rewards
        self.stockfish_metric = stockfish_metric

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        move_indices, board_tensors, end_rewards = translate.encode_seq(
            self.games[idx]["moves"],
            self.board_to_tensor,
        )
        input_dict = translate.format_inputs(
            move_indices,
            board_tensors,
            end_rewards,
            sequence=self.games[idx]["moves"],
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            device=self.device,
            window_size=self.window_size,
            generator=self.generator,
            return_dict=True,
            return_labels=self.eval_mode,
            shaping_rewards=self.shaping_rewards,
            stockfish_metric=self.stockfish_metric,
        )
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].squeeze(0)  # Remove batch dim
        if self.return_ids:
            input_dict["gameid"] = self.games[idx]["gameid"]
        return input_dict


class OnePlayerChessDataset(Dataset):
    def __init__(
        self,
        file_name: str,
        board_to_tensor: Optional[Callable[[chess.Board], torch.Tensor]],
        act_dim: int,
        state_dim: int,
        window_size: int,
        generator: torch.Generator,
        return_ids: bool = False,
        eval_mode: bool = False,
        shaping_rewards: bool = False,
        stockfish_metric: Optional[StockfishMetric] = None,
    ):
        self.games = []  # Can be heavy on memory, but good enough for now
        with jsonlines.open(file_name) as reader:
            self.games.extend(iter(reader))
        self.board_to_tensor = board_to_tensor
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.device = torch.device("cpu")  # Load full dataset on cpu
        self.window_size = window_size
        self.generator = generator
        self.return_ids = return_ids
        self.eval_mode = eval_mode
        self.shaping_rewards = shaping_rewards
        self.stockfish_metric = stockfish_metric

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        move_indices, board_tensors, end_rewards = translate.encode_seq(
            self.games[idx]["moves"],
            self.board_to_tensor,
        )
        input_dict = translate.format_inputs(
            move_indices,
            board_tensors,
            end_rewards,
            sequence=self.games[idx]["moves"],
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            device=self.device,
            window_size=self.window_size,
            generator=self.generator,
            return_dict=True,
            return_labels=self.eval_mode,
            one_player=True,
            shaping_rewards=self.shaping_rewards,
            stockfish_metric=self.stockfish_metric,
        )
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].squeeze(0)  # Remove batch dim
        if self.return_ids:
            input_dict["gameid"] = self.games[idx]["gameid"]
        return input_dict
