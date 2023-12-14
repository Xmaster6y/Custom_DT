"""
File implementing the inference wrapper class.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import chess
import torch

from src.utils import leela_encodings, translate


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
    position_evaluator: Callable[[chess.Board, Tuple[bool, bool]], float] = None


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


class LeelaInferenceWrapper:
    def __init__(
        self,
        model,
        config: InferenceConfig,
        color: chess.Color,
    ):
        self.model = model
        self.config = config
        self.color = color
        self.board_tensors = {
            chess.WHITE: [],
            chess.BLACK: [],
        }
        self.move_indices = {
            chess.WHITE: [],
            chess.BLACK: [],
        }
        self.end_rewards = {
            chess.WHITE: config.end_rewards[0],
            chess.BLACK: config.end_rewards[1],
        }
        if config.shaping_rewards:
            self.position_evaluations = {
                chess.WHITE: [],
                chess.BLACK: [],
            }
        else:
            self.position_evaluations = None
        if model.config.act_dim != 1858:
            raise NotImplementedError(f"The model has act_dim={model.config.act_dim}")
        if model.config.state_dim != 1280:
            raise NotImplementedError(f"The model has state_dim={model.config.state_dim}")
        self.board_to_tensor = leela_encodings.board_to_tensor

    def _format_inputs(self):
        """
        Format the inputs.
        """
        act_dim = self.model.config.act_dim
        state_dim = self.model.config.state_dim
        device = self.config.device

        colors = {0: chess.WHITE, 1: chess.BLACK}
        if not self.config.two_player:
            players = [0 if self.color == chess.WHITE else 1]
        else:
            players = [0, 1]

        seq_len = len(self.board_tensors[chess.WHITE])
        window_size = min(self.config.window_size, seq_len)
        window_start = seq_len - window_size

        states, actions, returns_to_go, attention_mask, timesteps = [], [], [], [], []
        for player in players:
            color = colors[player]
            player_states, player_actions, player_returns_to_go, player_attention_mask = leela_encodings.format_tensors(
                self.move_indices[color] + [0],
                self.board_tensors[color],
                self.end_rewards[color],
                window_size,
                device,
                window_start=window_start,
                position_evaluations=self.position_evaluations[color] if self.config.shaping_rewards else None,
                shaping_rewards=self.config.shaping_rewards,
            )
            player_timesteps = player + torch.arange(
                start=2 * window_start, end=2 * window_start + 2 * window_size, step=2, device=device
            ).unsqueeze(0)
            states.append(player_states.unsqueeze(2))
            actions.append(player_actions.unsqueeze(2))
            returns_to_go.append(player_returns_to_go.unsqueeze(2))
            attention_mask.append(player_attention_mask.unsqueeze(2))
            timesteps.append(player_timesteps.unsqueeze(2))
        n_players = len(players)
        states = torch.cat(states, dim=2).reshape(1, n_players * window_size, state_dim)
        actions = torch.cat(actions, dim=2).reshape(1, n_players * window_size, act_dim)
        returns_to_go = torch.cat(returns_to_go, dim=2).reshape(1, n_players * window_size, 1)
        attention_mask = torch.cat(attention_mask, dim=2).reshape(1, n_players * window_size)
        timesteps = torch.cat(timesteps, dim=2).reshape(1, n_players * window_size)

        return {
            "states": states,
            "actions": actions,
            "returns_to_go": returns_to_go,
            "attention_mask": attention_mask,
            "timesteps": timesteps,
        }

    def __call__(self, board: chess.Board):
        """
        Make a move.
        """
        us = self.color
        them = not us
        if self.config.two_player:
            try:
                last_move = board.pop()
                self.board_tensors[them].append(self.board_to_tensor(board, (them, us)))
                if self.config.shaping_rewards:
                    self.position_evaluations[them].append(self.config.position_evaluator(board, (them, us)))
                self.move_indices[them].append(leela_encodings.encode_move(last_move, (them, us)))
                board.push(last_move)
            except IndexError:
                pass
        self.board_tensors[us].append(self.board_to_tensor(board, (us, them)))
        if self.config.shaping_rewards:
            self.position_evaluations[us].append(self.config.position_evaluator(board, (us, them)))
        input_dict = self._format_inputs()
        output_dict = self.model(**input_dict)
        logits = output_dict["action_preds"][0, -1, :]
        top_k_indices = torch.topk(logits, self.config.top_k).indices
        top_k_moves = [leela_encodings.decode_move(idx.item(), (us, them), board) for idx in top_k_indices]
        for move, move_idx in zip(top_k_moves, top_k_indices):
            if move in board.legal_moves:
                break
            if self.config.debug:
                print(f"Move {move} is not legal.")
        else:
            raise RuntimeError(f"No legal move found in top {self.config.top_k} moves.")
        self.move_indices[us].append(move_idx)
        return move
