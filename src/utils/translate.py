"""
Encoding functions for converting between different representations of the board.
"""

import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import chess
import torch

from src.metric.stockfish import StockfishMetric


def encode_seq(
    seq: str,
    board_to_tensor: Optional[Callable[[chess.Board], torch.Tensor]] = None,
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
    sequence: str,
    act_dim: int,
    state_dim: int,
    device: torch.device,
    window_size: int,
    generator: torch.Generator,
    return_dict: bool = False,
    return_labels: bool = False,
    one_player: bool = False,
    shaping_rewards: bool = False,
    stockfish_metric: Optional[StockfishMetric] = None,
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
    black_returns = torch.ones((1, black_seq_len, 1), device=device) * end_rewards[1]
    white_returns = torch.ones((1, white_seq_len, 1), device=device) * end_rewards[0]

    if shaping_rewards:
        if stockfish_metric is None:
            raise ValueError("Stockfish metric must be provided if shaping rewards are enabled.")
        eval_list = [0] + stockfish_metric.eval_sequence(sequence, player="both", evaluation_depth=8)[:-1]
        evaluations = torch.tensor([eval_list])[:, :, None]

        white_returns = white_returns + evaluations[:, ::2, :]
        black_returns = black_returns + evaluations[:, 1::2, :]
        # evaluations are added instead of subtracted, because evaluations are from the
        # perspective of the player who has just moved. Needs to be perspective of player who is about to move.

    condition = torch.arange(seq_len, device=device) % 2 == 0
    returns_to_go = torch.zeros(1, seq_len, 1, device=device, dtype=torch.float32)
    returns_to_go[:, condition, :] = white_returns
    returns_to_go[:, ~condition, :] = black_returns
    returns_to_go = returns_to_go[:, window_start : window_start + window_size, :]

    timesteps = torch.arange(start=window_start, end=window_start + window_size, device=device).reshape(1, window_size)

    if one_player:
        color = torch.randint(2, (1,), generator=generator).item()
        states = states[:, color::2, :]
        actions = actions[:, color::2, :]
        returns_to_go = returns_to_go[:, color::2, :]
        timesteps = timesteps[:, color::2]

    if not return_dict:
        return states, actions, returns_to_go, timesteps
    input_dict = {
        "states": states,
        "actions": actions,
        "returns_to_go": returns_to_go,
        "timesteps": timesteps,
    }
    if return_labels:
        input_dict["labels"] = actions
    return input_dict


def decode_move(move_index: int, board: chess.Board) -> chess.Move:
    """
    Converts a move index to a chess.Move object.
    """
    if move_index < 4096:
        to_square, from_square = divmod(move_index, 64)
        mv = chess.Move(from_square, to_square)
        piece = board.piece_at(from_square)
        # from square is unoccupied
        if piece is None:
            return None
        else:
            to_rank = to_square // 8
            if piece.piece_type == chess.PAWN and to_rank in [0, 7]:
                mv.promotion = chess.QUEEN
            return mv

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


def index_to_piece(index: int):
    return "kqrbnp0PNBRQK"[index + 6]


def board_to_64tensor(board: chess.Board):
    """
    Converts a chess.Board object to a 64 tensor.
    """
    fen_rep = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_rep)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)

    return torch.tensor(tuple(map(piece_to_index, ordered_fen)), dtype=torch.int8)


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


def board_to_68tensor(board: chess.Board):
    """
    Converts a chess.Board object to a 68 tensor.
    64 squares + 4 castling rights
    """
    fen_rep = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_rep)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)
    ordinal_board = list(map(piece_to_index, ordered_fen))
    ordinal_board.append(board.has_kingside_castling_rights(chess.WHITE))
    ordinal_board.append(board.has_queenside_castling_rights(chess.WHITE))
    ordinal_board.append(board.has_kingside_castling_rights(chess.BLACK))
    ordinal_board.append(board.has_queenside_castling_rights(chess.BLACK))
    return torch.tensor(ordinal_board, dtype=torch.int8)


def board_to_72tensor(board: chess.Board):
    """
    Converts a chess.Board object to a 72 tensor.
    64 squares + 1 turn + 4 castling rights + 1 en passant square
        + 1 halfmove clock + 1 fullmove number
    Format: [a1, b1, c1,..., turn==white, white kingside CR, white queenside CR,
        black kingside CR, black queenside CR, en passant square index (or 0 if None),
        halfmove clock, fullmove number]
    """
    fen_rep = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_rep)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)
    ordinal_board = list(map(piece_to_index, ordered_fen))
    ordinal_board.append(board.turn)
    ordinal_board.append(board.has_kingside_castling_rights(chess.WHITE))
    ordinal_board.append(board.has_queenside_castling_rights(chess.WHITE))
    ordinal_board.append(board.has_kingside_castling_rights(chess.BLACK))
    ordinal_board.append(board.has_queenside_castling_rights(chess.BLACK))
    ordinal_board.append(board.ep_square if board.ep_square is not None else 0)
    ordinal_board.append(board.halfmove_clock)
    ordinal_board.append(board.fullmove_number)
    return torch.tensor(ordinal_board, dtype=torch.int16)


def complete_tensor_to_fen(board_tensor: torch.Tensor):
    rows = board_tensor[:64].reshape(8, 8)
    info = board_tensor[64:]
    rows = rows.flip(0)
    fen = ""
    for row_idx, row in enumerate(rows):
        empty = 0
        for square in row:
            if square == 0:
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += index_to_piece(square)
        if empty > 0:
            fen += str(empty)
        if row_idx < 7:
            fen += "/"
    fen += " w " if info[0] == 1 else " b "
    if info[1] == 1:
        fen += "K"
    if info[2] == 1:
        fen += "Q"
    if info[3] == 1:
        fen += "k"
    if info[4] == 1:
        fen += "q"
    if info[1] + info[2] + info[3] + info[4] == 0:
        fen += "-"
    if info[5] != 0:
        fen += " " + chess.square_name(info[5])
    else:
        fen += " -"
    fen += " " + str(info[6].item())
    fen += " " + str(info[7].item())
    return fen


def board_to_772tensor(board: chess.Board):
    """
    Converts a chess.Board object to a 772 tensor.
    Order of pieces: kqrbnpPNBRQK
    """
    board_64x12tensor = board_to_64x12tensor(board)
    board_772tensor = torch.zeros(772, dtype=torch.int8)
    board_772tensor[:768] = board_64x12tensor.flatten()
    board_772tensor[768] = board.has_kingside_castling_rights(chess.WHITE)
    board_772tensor[769] = board.has_queenside_castling_rights(chess.WHITE)
    board_772tensor[770] = board.has_kingside_castling_rights(chess.BLACK)
    board_772tensor[771] = board.has_queenside_castling_rights(chess.BLACK)
    return board_772tensor


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
