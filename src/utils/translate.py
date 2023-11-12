"""
Encoding functions for converting between different representations of the board.
"""

import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import chess
import torch


def encode_seq(
    seq: str, board_to_tensor: Optional[Callable[[chess.Board], torch.Tensor]] = None
) -> Tuple[List[int], Optional[List[torch.Tensor]], Tuple[float, float]]:
    """
    Converts a sequence of moves in algebraic notation to a sequence of move indices.
    """
    board = chess.Board()
    move_indices = []
    board_tensors = None if board_to_tensor is None else [board_to_tensor(board)]
    for alg_move in seq.split():
        if alg_move.endswith("."):
            continue
        move = board.push_san(alg_move)
        promotion = move.promotion
        if promotion is not None and promotion != chess.QUEEN:  # Underpromotion
            direction = (move.to_square % 8) - (move.from_square % 8)
            extra_index = (promotion - 2) + 3 * (direction + 1) + 9 * move.from_square
            move_indices.append(4096 + extra_index)
        else:
            move_indices.append(move.from_square + 64 * move.to_square)
        if board_to_tensor is not None:
            board_tensors.append(board_to_tensor(board))
    if board_to_tensor is not None:
        board_tensors.pop()  # Remove the last board tensor, since it is not needed

    outcome = board.outcome()
    if outcome is None:
        end_rewards = (0.0, 0.0)
    elif outcome.winner == chess.WHITE:
        end_rewards = (1.0, -1.0)
    elif outcome.winner == chess.BLACK:
        end_rewards = (-1.0, 1.0)
    else:
        end_rewards = (0.5, 0.5)

    return move_indices, board_tensors, end_rewards


def format_inputs(
    move_indices: List[int],
    board_tensors: List[torch.Tensor],
    end_rewards: Tuple[float, float],
    act_dim: int,
    state_dim: int,
    device: torch.device,
    discount: float,
    window_size: int,
    generator: torch.Generator,
    return_dict: bool = False,
    return_labels: bool = False,
    one_player: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Prepares the data for the model.
    """

    seq_len = len(move_indices)
    if window_size > seq_len:
        raise NotImplementedError("Window size must be less than or equal to the sequence length.")
        # TODO: Implement padding
    window_start = torch.randint(seq_len - window_size, (1,), generator=generator).item()

    action_seq = torch.nn.functional.one_hot(
        torch.tensor(move_indices[window_start : window_start + window_size], dtype=int), num_classes=act_dim
    )
    actions = action_seq.reshape(1, window_size, act_dim).to(device=device, dtype=torch.float32)

    state_seq = torch.stack(board_tensors[window_start : window_start + window_size])
    states = state_seq.reshape(1, window_size, state_dim).to(device=device, dtype=torch.float32)

    black_seq_len = seq_len // 2
    white_seq_len = seq_len - black_seq_len
    black_returns = discount ** torch.arange(black_seq_len, device=device) * end_rewards[1]
    white_returns = discount ** torch.arange(white_seq_len, device=device) * end_rewards[0]

    condition = torch.arange(seq_len, device=device) % 2 == 0
    returns_to_go = torch.zeros(1, seq_len, 1, device=device, dtype=torch.float32)
    returns_to_go[:, condition, :] = white_returns.reshape(1, white_seq_len, 1)
    returns_to_go[:, ~condition, :] = black_returns.reshape(1, black_seq_len, 1)
    returns_to_go = returns_to_go[:, window_start : window_start + window_size, :]

    timesteps = torch.arange(start=window_start, end=window_start + window_size, device=device).reshape(1, window_size)
    attention_mask = torch.ones(1, window_size, device=device, dtype=torch.float32)  # Needed for padding

    if one_player:
        color = torch.randint(2, (1,), generator=generator).item()
        states = states[:, color::2, :]
        actions = actions[:, color::2, :]
        returns_to_go = returns_to_go[:, color::2, :]
        timesteps = timesteps[:, color::2]
        attention_mask = attention_mask[:, color::2]

    if return_dict:
        input_dict = {
            "states": states,
            "actions": actions,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
        }
        if return_labels:
            input_dict["labels"] = actions
        return input_dict
    else:
        return states, actions, returns_to_go, timesteps, attention_mask


def decode_move(move_index: int) -> chess.Move:
    """
    Converts a move index to a chess.Move object.
    """
    if move_index < 4096:
        from_square = move_index % 64
        to_square = move_index // 64
        return chess.Move(from_square, to_square)
    else:
        extra_index = move_index - 4096
        promotion = extra_index % 3 + 2
        extra_index = extra_index // 3
        direction = extra_index % 3 - 1
        from_square = extra_index // 3
        from_rank = from_square // 8
        from_file = from_square % 8
        to_rank = 7 if from_rank == 6 else 0
        to_file = from_file + direction
        to_square = to_file + 8 * to_rank
        return chess.Move(from_square, to_square, promotion)


def piece_to_index(piece: str):
    return "kqrbnp0PNBRQK".index(piece) - 6


def board_to_64tensor(board: chess.Board):
    """
    Converts a chess.Board object to a 64 tensor.
    """
    fen_rep = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_rep)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)

    # Convert to a 64 tensor
    board_tensor = torch.tensor(tuple(map(piece_to_index, ordered_fen)), dtype=torch.int8)
    return board_tensor


def board_to_64x12tensor(board: chess.Board):
    """
    Converts a chess.Board object to a 64x12 tensor.
    Order of pieces: kqrbnpPNBRQK
    """
    board_64tensor = board_to_64tensor(board)
    board_64x12tensor = torch.zeros(64, 12, dtype=torch.int8)
    for piece_index in range(1, 7):
        board_64x12tensor[:, piece_index + 5] = board_64tensor == piece_index
        board_64x12tensor[:, 6 - piece_index] = board_64tensor == -piece_index
    return board_64x12tensor.flatten()


if __name__ == "__main__":
    seq = "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"
    print(f"Sequence: {seq}")
    print(f"Decoded: {encode_seq(seq)}")
    print(f"Black queen: {piece_to_index('q')}")
    print(f"White bishop: {piece_to_index('B')}")
    print(f"Empty square: {piece_to_index('0')}")
    board = chess.Board()
    print(f"Board: {board}")
    print(f"Board tensor: {board_to_64tensor(board)}")
    print(f"Board tensor: {board_to_64x12tensor(board)}")
