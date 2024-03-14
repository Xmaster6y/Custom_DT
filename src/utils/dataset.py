"""
Dataset class for loading data from jsonl files.

Typical usage example:
```python
>>> data = LeelaChessDataset(
        file_name="data/chess_games_base/test_stockfish_5000.jsonl",
        position_evaluator=position_evaluator,
        window_size=args.window_size,
        generator=eval_generator,
        eval_mode=True,
        one_player=args.one_player,
    )
>>> data[0]
```
"""

from typing import Callable, Optional

import chess
import jsonlines
import torch
from torch.utils.data import Dataset

from src.metric.stockfish import StockfishMetric
from src.utils import leela_encodings, translate


class TwoPlayersChessDataset(Dataset):
    """
    Loads dataset from json file and preserves both players' moves.

    This dataset uses deprecated encodings. Use LeelaChessDataset instead.

    Attributes:
        games: List of games loaded from json file.
        board_to_tensor: Function that converts a chess.Board to a torch.Tensor.
        act_dim: Number of possible actions in the dataset.
        state_dim: Dimension of the state space.
        device: Device to load data on.
        window_size: Number of moves to consider for each sample.
        generator: Generator to use for random sampling.
        return_ids: Whether to return the game id with the sample.
        eval_mode: Whether to return the labels.
        shaping_rewards: Whether to return the shaping rewards.
        stockfish_metric: StockfishMetric object to use for stockfish evaluation shaping rewards.
    """

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
        """
        Initializes a TwoPlayersChessDataset object.

        Args:
            file_name: Path to jsonl file.
            board_to_tensor: Function that converts a chess.Board to a torch.Tensor.
            act_dim: Number of possible actions in the dataset.
            state_dim: Dimension of the state space.
            window_size: Number of moves to consider for each sample.
            generator: Generator to use for random sampling.
            return_ids: Whether to return the game id with the sample.
            eval_mode: Whether to return the labels.
            shaping_rewards: Whether to return the shaping rewards.
            stockfish_metric: StockfishMetric object to use for stockfish evaluation shaping rewards.
        """
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
        """
        Formats a game from the dataset into a dictionary of tensors.

        Args:
            idx: Index of the game to format.

        Returns:
            Dictionary of tensors containing the formatted game.
        """
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
    """
    Loads dataset from json file and preserves only one player's moves.

    The player is chosen randomly for each sample. This dataset uses deprecated encodings.
    Use LeelaChessDataset instead.

    Attributes:
        games: List of games loaded from json file.
        board_to_tensor: Function that converts a chess.Board to a torch.Tensor.
        act_dim: Number of possible actions in the dataset.
        state_dim: Dimension of the state space.
        device: Device to load data on.
        window_size: Number of moves to consider for each sample.
        generator: Generator to use for random sampling.
        return_ids: Whether to return the game id with the sample.
        eval_mode: Whether to return the labels.
        shaping_rewards: Whether to return the shaping rewards.
        stockfish_metric: StockfishMetric object to use for stockfish evaluation shaping rewards.
    """

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
        """
        Initializes a OnePlayerChessDataset object.

        Args:
            file_name: Path to jsonl file.
            board_to_tensor: Function that converts a chess.Board to a torch.Tensor.
            act_dim: Number of possible actions in the dataset.
            state_dim: Dimension of the state space.
            window_size: Number of moves to consider for each sample.
            generator: Generator to use for random sampling.
            return_ids: Whether to return the game id with the sample.
            eval_mode: Whether to return the labels.
            shaping_rewards: Whether to return the shaping rewards.
            stockfish_metric: StockfishMetric object to use for stockfish evaluation shaping rewards.
        """
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
        """
        Formats a game from the dataset into a dictionary of tensors.

        Args:
            idx: Index of the game to format.

        Returns:
            Dictionary of tensors containing the formatted game.
        """
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


class LeelaChessDataset(Dataset):
    """
    Loads dataset from json file and encodes the games using Leela encodings.

    Attributes:
        games: List of games loaded from json file.
        device: Device to load data on.
        window_size: Number of moves to consider for each sample.
        generator: Generator to use for random sampling.
        eval_mode: Whether to return the labels.
        position_evaluator: Function that evaluates a chess.Board position.
        one_player: Whether to encode the data using only one player's moves.
        pad_token_id: Value to use for padding.
    """

    def __init__(
        self,
        file_name: str,
        window_size: int,
        generator: torch.Generator,
        eval_mode: bool = False,
        position_evaluator: Optional[Callable[[chess.Board], float]] = None,
        one_player: bool = False,
        pad_token_id: int = 0,
    ):
        """
        Initializes a LeelaChessDataset object.

        Args:
            file_name: Path to jsonl file.
            window_size: Number of moves to consider for each sample.
            generator: Generator to use for random sampling.
            eval_mode: Whether to return the labels.
            position_evaluator: Function that evaluates a chess.Board position.
            one_player: Whether to encode the data using only one player's moves.
            pad_token_id: Value to use for padding.
        """
        self.games = []  # Can be heavy on memory, but good enough for now
        with jsonlines.open(file_name) as reader:
            self.games.extend(iter(reader))
        self.device = torch.device("cpu")  # Load full dataset on cpu
        self.window_size = window_size
        self.generator = generator
        self.eval_mode = eval_mode
        self.position_evaluator = position_evaluator
        self.one_player = one_player
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        """
        Formats a game from the dataset into a dictionary of tensors.

        Args:
            idx: Index of the game to format.

        Returns:
            Dictionary of tensors containing the formatted game.
        """
        encoded_seq = leela_encodings.encode_seq(
            seq=self.games[idx]["moves"],
            board_to_tensor=leela_encodings.board_to_tensor,
            move_to_index=leela_encodings.encode_move,
            return_last_board=False,
            position_evaluator=self.position_evaluator,
        )
        input_dict = leela_encodings.format_inputs(
            encoded_seq=encoded_seq,
            device=self.device,
            window_size=self.window_size,
            generator=self.generator,
            return_labels=self.eval_mode,
            one_player=self.one_player,
            shaping_rewards=self.position_evaluator is not None,
            pad_token_id=self.pad_token_id,
        )
        for key in input_dict:
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].squeeze(0)  # Remove batch dim
        return input_dict
