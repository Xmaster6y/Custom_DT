"""
File implementing the inference wrapper class.
"""
from dataclasses import dataclass

import chess
import torch

from src.utils import translate


@dataclass
class InferenceConfig:
    """
    Configuration class for the inference wrapper.
    """

    two_player: bool = True
    window_size: int = 10
    shaping_rewards: bool = False
    device: torch.device = torch.device("cpu")
    end_rewards: tuple = (1.0, -1.0)
    top_k: int = 5
    debug: bool = True


class InferenceWrapper:
    def __init__(
        self,
        model,
        config: InferenceConfig,
    ):
        self.model = model
        self.config = config
        self.board_tensors = []
        self.move_indices = []
        if model.config.act_dim != 4672:
            raise NotImplementedError("The model must have an action dimension of 4672.")
        if model.config.state_dim == 64:
            self.board_to_tensor = translate.board_to_64tensor
        elif model.config.state_dim == 768:
            self.board_to_tensor = translate.board_to_64x12tensor
        else:
            raise NotImplementedError("The model must have a state dimension of 64 or 768.")

    def _format_inputs(self):
        """
        Format the inputs.
        """
        act_dim = self.model.config.act_dim
        state_dim = self.model.config.state_dim
        device = self.config.device

        seq_len = len(self.board_tensors)
        window_size = min(self.config.window_size, seq_len)
        window_start = seq_len - window_size

        action_seq = torch.nn.functional.one_hot(
            torch.tensor(self.move_indices[window_start : window_start + window_size] + [0], dtype=int),
            num_classes=act_dim,
        )
        actions = action_seq.reshape(1, window_size, act_dim).to(device=device, dtype=torch.float32)

        state_seq = torch.stack(self.board_tensors[window_start : window_start + window_size])
        states = state_seq.reshape(1, window_size, state_dim).to(device=device, dtype=torch.float32)

        black_seq_len = seq_len // 2
        white_seq_len = seq_len - black_seq_len
        black_returns = torch.ones((1, black_seq_len, 1), device=device) * self.config.end_rewards[1]
        white_returns = torch.ones((1, white_seq_len, 1), device=device) * self.config.end_rewards[0]

        if self.config.shaping_rewards:
            raise NotImplementedError("Shaping rewards are not implemented yet.")

        condition = torch.arange(seq_len, device=device) % 2 == 0
        returns_to_go = torch.zeros(1, seq_len, 1, device=device, dtype=torch.float32)
        returns_to_go[:, condition, :] = white_returns
        returns_to_go[:, ~condition, :] = black_returns
        returns_to_go = returns_to_go[:, window_start : window_start + window_size, :]

        timesteps = torch.arange(start=window_start, end=window_start + window_size, device=device).reshape(
            1, window_size
        )

        return {
            "states": states,
            "actions": actions,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
        }

    def __call__(self, board: chess.Board):
        """
        Make a move.
        """
        if self.config.two_player:
            try:
                last_move = board.pop()
                self.board_tensors.append(self.board_to_tensor(board))
                self.move_indices.append(translate.encode_move(last_move))
                board.push(last_move)
            except IndexError:
                pass
        self.board_tensors.append(self.board_to_tensor(board))
        input_dict = self._format_inputs()
        output_dict = self.model(**input_dict)
        logits = output_dict["action_preds"][0, -1, :]
        top_k_indices = torch.topk(logits, self.config.top_k).indices
        top_k_moves = [translate.decode_move(idx.item()) for idx in top_k_indices]
        for move, move_idx in zip(top_k_moves, top_k_indices):
            if move in board.legal_moves:
                break
            if self.config.debug:
                print(f"Move {move} is not legal.")
        else:
            raise RuntimeError(f"No legal move found in top {self.config.top_k} moves.")
        self.move_indices.append(move_idx)
        return move

    def reset(self):
        """
        Reset the inference wrapper.
        """
        self.board_tensors = []
        self.move_indices = []

    def pop(self):
        """
        Pop a move.
        """
        self.board_tensors.pop()
        self.move_indices.pop()
