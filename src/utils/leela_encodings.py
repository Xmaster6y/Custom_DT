"""
File containing the encodings for the Leela Chess Zero engine.
"""

import re
from copy import deepcopy
from typing import Callable, Tuple

import chess
import torch

from src.utils.leela_constants import INVERTED_POLICY_INDEX, POLICY_INDEX


def board_to_tensor13x8x8(
    board: chess.Board,
    us_them: Tuple[bool, bool],
    repetition: int = 2,
):
    """
    Converts a chess.Board object to a 13x8x8 tensor.
    """
    us, them = us_them
    plane_orders = {chess.WHITE: "PNBRQK", chess.BLACK: "pnbrqk"}
    plane_order = plane_orders[us] + plane_orders[them]

    fen_rep = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_rep)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)
    tensor13x8x8 = torch.zeros((13, 8, 8), dtype=torch.float)
    for i, piece in enumerate(plane_order):
        tensor13x8x8[i] = torch.tensor([[1 if c == piece else 0 for c in row] for row in ordered_fen]).view(8, 8)
    if board.is_repetition(repetition):
        tensor13x8x8[12] = torch.ones((8, 8), dtype=torch.float)
    return tensor13x8x8


def board_to_tensor112x8x8(last_board: chess.Board, us_them: Tuple[bool, bool]):
    """
    Create the lc0 112x8x8 tensor from the history of a game.
    """
    board = deepcopy(last_board)
    tensor112x8x8 = torch.zeros((112, 8, 8), dtype=torch.int8)
    us, them = us_them
    for i in range(8):
        tensor13x8x8 = board_to_tensor13x8x8(board, (us, them))
        if us == chess.BLACK:
            tensor13x8x8 = tensor13x8x8.flip(1)
        tensor112x8x8[i * 13 : (i + 1) * 13] = tensor13x8x8
        try:
            board.pop()
        except IndexError:
            break
    if last_board.has_queenside_castling_rights(us):
        tensor112x8x8[104] = torch.ones((8, 8), dtype=torch.int8)
    if last_board.has_kingside_castling_rights(us):
        tensor112x8x8[105] = torch.ones((8, 8), dtype=torch.int8)
    if last_board.has_queenside_castling_rights(them):
        tensor112x8x8[106] = torch.ones((8, 8), dtype=torch.int8)
    if last_board.has_kingside_castling_rights(them):
        tensor112x8x8[107] = torch.ones((8, 8), dtype=torch.int8)
    if us == chess.BLACK:
        tensor112x8x8[108] = torch.ones((8, 8), dtype=torch.int8)
    if last_board.is_fifty_moves():
        tensor112x8x8[109] = torch.ones((8, 8), dtype=torch.int8)
    tensor112x8x8[111] = torch.ones((8, 8), dtype=torch.int8)
    return tensor112x8x8


def board_to_tensor(
    board: chess.Board,
    us_them: Tuple[bool, bool],
    num_past_states: int = 0,
    flip: bool = True,
):
    """
    Converts a chess.Board object to a tensor.
    """
    if num_past_states > 0:
        raise NotImplementedError("This function does not support past states.")
    last_board = board.copy()
    us, them = us_them
    tensor13x8x8 = board_to_tensor13x8x8(board, (us, them))
    if flip and them == chess.WHITE:
        tensor13x8x8 = tensor13x8x8.flip(1, 2)
    board_planes = 13 * (num_past_states + 1)
    extra_planes = 7
    final_tensor = torch.zeros((board_planes + extra_planes, 8, 8), dtype=torch.float)
    if last_board.has_queenside_castling_rights(us):
        final_tensor[board_planes] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_kingside_castling_rights(us):
        final_tensor[board_planes + 1] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_queenside_castling_rights(them):
        final_tensor[board_planes + 2] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_kingside_castling_rights(them):
        final_tensor[board_planes + 3] = torch.ones((8, 8), dtype=torch.float)
    if us == chess.BLACK:
        final_tensor[board_planes + 4] = torch.ones((8, 8), dtype=torch.float)
    final_tensor[board_planes + 5] = torch.ones((8, 8), dtype=torch.float) * last_board.halfmove_clock / 100.0
    final_tensor[board_planes + 6] = torch.ones((8, 8), dtype=torch.float)
    return final_tensor


def encode_move(
    move: chess.Move,
    us_them: Tuple[bool, bool],
) -> int:
    """
    Converts a chess.Move object to an index.
    """
    us, _ = us_them
    from_square = move.from_square
    to_square = move.to_square

    if us == chess.BLACK:
        from_square = 63 - from_square
        to_square = 63 - to_square
    us_uci_move = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]
    if move.promotion is not None:
        if move.promotion == chess.BISHOP:
            us_uci_move += "b"
        elif move.promotion == chess.ROOK:
            us_uci_move += "r"
        elif move.promotion == chess.QUEEN:
            us_uci_move += "q"
        # Knight promotion is the default
    return INVERTED_POLICY_INDEX[us_uci_move]


def decode_move(
    index: int,
    us_them: Tuple[bool, bool],
    board: chess.Board = chess.Board(),
) -> chess.Move:
    """
    Converts an index to a chess.Move object.
    """
    us, _ = us_them
    us_uci_move = POLICY_INDEX[index]
    from_square = chess.SQUARE_NAMES.index(us_uci_move[:2])
    to_square = chess.SQUARE_NAMES.index(us_uci_move[2:4])
    if us == chess.BLACK:
        from_square = 63 - from_square
        to_square = 63 - to_square
    uci_move = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]
    from_piece = board.piece_at(from_square)
    if from_piece == chess.PAWN and to_square >= 56:  # Pawn promotion to knight by default
        uci_move += "n"
    return chess.Move.from_uci(uci_move)


def encode_seq(
    seq: str,
    board_to_tensor: Callable[[chess.Board], torch.Tensor],
    move_to_index: Callable[[chess.Move], int],
    return_last_board: bool = False,
    move_evaluator: Callable[[chess.Board], float] = None,
):
    """
    Converts a sequence of moves in algebraic notation to a sequence of move indices.
    """
    board = chess.Board()
    move_indices = {
        chess.WHITE: [],
        chess.BLACK: [],
    }
    board_tensors = {
        chess.WHITE: [],
        chess.BLACK: [],
    }
    move_evaluations = {
        chess.WHITE: [],
        chess.BLACK: [],
    }
    us, them = chess.WHITE, chess.BLACK
    for alg_move in seq.split():
        board_tensors[us].append(board_to_tensor(board, (us, them)))
        if alg_move.endswith("."):
            continue
        move = board.push_san(alg_move)
        move_indices[us].append(move_to_index(move, (us, them)))
        if move_evaluator is not None:
            move_evaluations[us].append(move_evaluator(board))
        us, them = them, us

    outcome = board.outcome()
    if outcome is None:
        end_rewards = (0.0, 0.0)
    elif outcome.winner == chess.WHITE:
        end_rewards = (1.0, -1.0)
    elif outcome.winner == chess.BLACK:
        end_rewards = (-1.0, 1.0)
    else:
        end_rewards = (0.5, 0.5)
    end_rewards = {
        chess.WHITE: end_rewards[0],
        chess.BLACK: end_rewards[1],
    }
    return {
        "move_indices": move_indices,
        "board_tensors": board_tensors,
        "end_rewards": end_rewards,
        "last_board": board if return_last_board else None,
    }
