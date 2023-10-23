"""
Dataset class for loading data from jsonl files.
"""

from typing import Callable, Optional

import chess
import jsonlines
import torch
from torch.utils.data import Dataset, default_collate

import src.utils.translate as translate


def custom_collate_fn(batch: dict) -> dict:
    """
    Custom collate function for the chess dataset.
    """
    batch = default_collate(batch)
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].squeeze(1)
    return batch


class ChessDataset(Dataset):
    def __init__(
        self,
        file_name: str,
        board_to_tensor: Optional[Callable[[chess.Board], torch.Tensor]],
        act_dim: int,
        state_dim: int,
        discount: float,
        window_size: int,
        generator: torch.Generator,
        return_ids: bool = False,
    ):
        self.games = []  # Can be heavy on memory, but good enough for now
        with jsonlines.open(file_name) as reader:
            for obj in reader:
                self.games.append(obj)
        self.board_to_tensor = board_to_tensor
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.device = torch.device("cpu")  # Load full dataset on cpu
        self.discount = discount
        self.window_size = window_size
        self.generator = generator
        self.return_ids = return_ids

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
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            device=self.device,
            discount=self.discount,
            window_size=self.window_size,
            generator=self.generator,
            return_dict=True,
        )
        if self.return_ids:
            input_dict["gameid"] = self.games[idx]["gameid"]
        return input_dict
