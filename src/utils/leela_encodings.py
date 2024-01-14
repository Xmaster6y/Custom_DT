"""
Encodes a dataset of chess games into a form suitable for the Leela Chess Zero engine.

These are the primary encodings used throughout the project.
"""

import re
from typing import Callable, Dict, Tuple

import chess
import torch

from src.utils.leela_constants import ACT_DIM, INVERTED_POLICY_INDEX, POLICY_INDEX, STATE_DIM


def board_to_tensor13x8x8(
    board: chess.Board,
    us_them: Tuple[bool, bool],
    repetition: int = 2,
) -> torch.Tensor:
    """
    Converts a chess.Board object to a 13x8x8 tensor.

    The first 12 planes are the ordinal positions of the pieces. The 13th plane is 1 if the position
    has been repeated n or more times, 0 otherwise. This board representation does not exlicitly
    encode castling rights, en passant, or the halfmove clock.

    Args:
        board: The chess.Board object to convert.
        us_them: A tuple of booleans indicating whether the board is from the perspective of the white player.
            (True, False) would represent a game from the perspective of the white player and vice versa.
        repetition: The number of times the position must be repeated for the 13th plane to be 1.

    Returns:
        A 13x8x8 tensor representing the board.

    Typical usage example:
    ```python
    >>> import chess
    >>> from src.utils.leela_encodings import board_to_tensor13x8x8
    >>> board = chess.Board()
    >>> board_to_tensor13x8x8(board, (True, False))
    ```
    """
    us, them = us_them
    plane_orders = {chess.WHITE: "PNBRQK", chess.BLACK: "pnbrqk"}
    plane_order = plane_orders[us] + plane_orders[them]

    def piece_to_index(piece: str):
        return f"{plane_order}0".index(piece)

    fen_rep = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_rep)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)

    tensor13x8x8 = torch.zeros((13, 8, 8), dtype=torch.float)
    ordinal_board = torch.tensor(tuple(map(piece_to_index, ordered_fen)), dtype=torch.float)
    ordinal_board = ordinal_board.reshape((8, 8)).unsqueeze(0)
    piece_tensor = torch.tensor(tuple(map(piece_to_index, plane_order)), dtype=torch.float)
    piece_tensor = piece_tensor.reshape((12, 1, 1))
    tensor13x8x8[:12] = (ordinal_board == piece_tensor).float()
    if board.is_repetition(repetition):
        tensor13x8x8[12] = torch.ones((8, 8), dtype=torch.float)
    return tensor13x8x8


def board_to_tensor(
    board: chess.Board,
    us_them: Tuple[bool, bool],
    num_past_states: int = 0,
    flip: bool = True,
) -> torch.Tensor:
    """
    Converts a chess.Board object to a tensor.

    The first 12 planes are the ordinal positions of the pieces. The 13th plane is 1 if the position has been
    repeated n or more times, 0 otherwise. The 14th-17th planes encode castling rights. The player to move
    is encoded in the 18th plane. The 19th plane is the halfmove clock.

    Args:
        board: The chess.Board object to convert.
        us_them: A tuple of booleans indicating whether the board is from the perspective of the white player.
            (True, False) would represent a game from the perspective of the white player and vice versa.
        num_past_states: The number of past states to include in the tensor. Not implemented.
        flip: Whether to flip the board if the black player is to move.

    Returns:
        A 20x8x8 tensor representing the board.

    Raises:
        NotImplementedError: If num_past_states > 0.

    Typical usage example:
    ```python
    >>> import chess
    >>> from src.utils.leela_encodings import board_to_tensor
    >>> board = chess.Board()
    >>> board_to_tensor(board, (True, False))
    ```
    """
    if num_past_states > 0:
        raise NotImplementedError("This function does not support past states.")
    last_board = board.copy()
    us, them = us_them
    tensor13x8x8 = board_to_tensor13x8x8(board, (us, them))
    if flip and them == chess.WHITE:
        tensor13x8x8 = tensor13x8x8.flip(1)
    board_planes = 13 * (num_past_states + 1)
    extra_planes = 7
    final_tensor = torch.zeros((board_planes + extra_planes, 8, 8), dtype=torch.float)
    final_tensor[:13] = tensor13x8x8
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
    final_tensor[board_planes + 5] = torch.ones((8, 8), dtype=torch.float) * last_board.halfmove_clock / 99.0
    final_tensor[board_planes + 6] = torch.ones((8, 8), dtype=torch.float)
    return final_tensor


def encode_move(
    move: chess.Move,
    us_them: Tuple[bool, bool],
) -> int:
    """
    Converts a chess.Move object to an index in the 1858-dimensional action space.

    Args:
        move: The chess.Move object to convert.
        us_them: A tuple of booleans indicating whether the move is from the perspective of the white player.
            (True, False) would represent a game from the perspective of the white player and vice versa.

    Returns:
        An integer index in the 1858-dimensional action space.

    Typical usage example:
    ```python
    >>> import chess
    >>> from src.utils.leela_encodings import encode_move
    >>> move = chess.Move.from_uci("e2e4")
    >>> encode_move(move, (True, False))
    ```
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

    Args:
        index: The index to convert.
        us_them: A tuple of booleans indicating whether the move is from the perspective of the white player.
            (True, False) would represent a game from the perspective of the white player and vice versa.
        board: The current chess.Board object from which the move is being made.

    Returns:
        A chess.Move object.

    Typical usage example:
    ```python
    >>> import chess
    >>> from src.utils.leela_encodings import decode_move
    >>> move_idx = 293
    >>> board = chess.Board()
    >>> decode_move(move_idx, (True, False), board)
    ```
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
    board_to_tensor: Callable[[chess.Board, Tuple[bool, bool]], torch.Tensor],
    move_to_index: Callable[[chess.Move, Tuple[bool, bool]], int],
    return_last_board: bool = False,
    position_evaluator: Callable[[chess.Board, Tuple[bool, bool]], float] = None,
) -> Dict[str, dict]:
    """
    Converts a sequence of moves in algebraic notation to a dictionary encoding the moves, the board states
    generated by the moves, the end rewards of the game, the Stockfish evaluations of the board states, and
    optionally the last board.

    Args:
        seq: The sequence of moves in algebraic notation.
        board_to_tensor: A function that converts a chess.Board object to a tensor.
        move_to_index: A function that converts a chess.Move object to an index in the action space.
        return_last_board: Whether to return the last board state.
        position_evaluator: A function that provides the Stockfish evaluation of a board position.

    Returns:
        A dictionary of dictionaries of white and black move tensors, board states, end rewards, Stockfish evaluations,
        and optionally the last board.

    Typical usage example:
    ```python
    >>> from src.utils.leela_encodings import encode_seq
    >>> from src.utils.leela_constants import board_to_tensor, encode_move
    >>> seq = "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"
    >>> encode_seq(seq, board_to_tensor, encode_move)
    ```
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
    position_evaluations = {
        chess.WHITE: [],
        chess.BLACK: [],
    }
    us, them = chess.WHITE, chess.BLACK
    for alg_move in seq.split():
        if alg_move.endswith("."):
            continue
        board_tensors[us].append(board_to_tensor(board, (us, them)))
        if position_evaluator is not None:
            position_evaluations[us].append(position_evaluator(board, (us, them)))
        move = board.push_san(alg_move)
        move_indices[us].append(move_to_index(move, (us, them)))
        us, them = them, us

    outcome = board.outcome()
    if outcome is None or outcome.winner not in [chess.WHITE, chess.BLACK]:
        end_rewards = (0.5, 0.5)
    elif outcome.winner == chess.WHITE:
        end_rewards = (1.0, -1.0)
    else:
        end_rewards = (-1.0, 1.0)
    end_rewards = {
        chess.WHITE: end_rewards[0],
        chess.BLACK: end_rewards[1],
    }
    return {
        "move_indices": move_indices,
        "board_tensors": board_tensors,
        "end_rewards": end_rewards,
        "position_evaluations": position_evaluations,
        "last_board": board if return_last_board else None,
    }


def format_tensors(
    move_indices: list,
    board_tensors: list,
    end_reward: tuple,
    window_size: int,
    device: torch.device,
    window_start: int = 0,
    position_evaluations: list = None,
    shaping_rewards: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Formats chess game sequence data into tensors suitable for modeling with the Decision Transformer.
    Primarily used by the format_inputs function.

    Args:
        move_indices: A list of move indices.
        board_tensors: A list of board tensors.
        end_reward: A tuple of end rewards for the white and black players.
        window_size: The length of the sequence to model.
        device: The device to store the tensors on.
        window_start: The starting index of the sequence to model.
        position_evaluations: A list of Stockfish evaluations of the board positions.
        shaping_rewards: Whether to use Stockfish evaluation shaping rewards.

    Returns:
        A tuple of tensors for modeling with the Decision Transformer.

    Raises:
        ValueError: If shaping rewards but either a.) the position evaluator provided to the encode_seq function has
            an error or b.) the position evaluator provided to the encode_seq function is None.
    """
    seq_len = len(move_indices)
    if window_start + window_size > seq_len:
        window_remainder = window_start + window_size - seq_len
        window_end = seq_len
    else:
        window_remainder = 0
        window_end = window_start + window_size
    attention_mask = torch.zeros((1, window_size), dtype=torch.float32)
    attention_mask[:, : window_size - window_remainder] = 1.0

    action_seq = torch.nn.functional.one_hot(
        torch.tensor(move_indices[window_start:window_end] + [0] * window_remainder, dtype=int), num_classes=ACT_DIM
    )
    actions = action_seq.reshape(1, window_size, ACT_DIM).to(device=device, dtype=torch.float32)

    state_seq = torch.stack(board_tensors[window_start:window_end])
    states = state_seq.reshape(1, window_end - window_start, STATE_DIM).to(device=device, dtype=torch.float32)
    states = torch.cat((states, torch.zeros((1, window_remainder, STATE_DIM), dtype=torch.float32)), dim=1)

    returns_to_go = torch.zeros(1, window_size, 1, device=device, dtype=torch.float32)
    returns_to_go[:, : window_size - window_remainder, :] = torch.full(
        (1, window_size - window_remainder, 1), end_reward, device=device, dtype=torch.float32
    )

    if shaping_rewards:
        if position_evaluations is None:
            raise ValueError("No move evaluations provided.")
        evaluations = torch.tensor(
            [position_evaluations[window_start:window_end] + [0.0] * window_remainder],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(2)
        returns_to_go = returns_to_go - evaluations

    return states, actions, returns_to_go, attention_mask


def format_inputs(
    encoded_seq: dict,
    device: torch.device,
    window_size: int,
    generator: torch.Generator,
    return_labels: bool = False,
    one_player: bool = True,
    shaping_rewards: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Converts an encoded sequence dictionary to a dictionary of tensors suitable for modeling with the Decision
    Transformer. This dictionary stores the states, actions, returns to go, attention masks, and timesteps
    of the sequence.

    Args:
        encoded_seq: The encoded sequence dictionary.
        device: The device to store the tensors on.
        window_size: The length of the sequence to model.
        generator: A torch.Generator object to use for random number generation.
        return_labels: Whether to return the actions as labels.
        one_player: Whether to model one player or both players.
        shaping_rewards: Whether to use Stockfish evaluation shaping rewards.

    Returns:
        A dictionary of tensors for modeling with the Decision Transformer.

    Typical usage example:
    ```python
    >>> import torch
    >>> from src.utils.leela_encodings import encode_seq, format_inputs
    >>> from src.utils.leela_constants import board_to_tensor, encode_move
    >>> seq = "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"
    >>> encoded_seq = encode_seq(seq, board_to_tensor, encode_move)
    >>> format_inputs(encoded_seq, torch.device("cpu"), 10, torch.Generator())
    ```
    """
    move_indices = encoded_seq["move_indices"]
    board_tensors = encoded_seq["board_tensors"]
    end_rewards = encoded_seq["end_rewards"]
    position_evaluations = encoded_seq["position_evaluations"]

    colors = {0: chess.WHITE, 1: chess.BLACK}
    if one_player:
        color_ind = torch.randint(2, (1,), generator=generator).item()
        players = [color_ind]
    else:
        players = [0, 1]

    seq_len = len(move_indices[colors[0]])
    if window_size >= seq_len:
        window_start = 0
    else:
        window_start = torch.randint(seq_len - window_size, (1,), generator=generator).item()
    states, actions, returns_to_go, attention_mask, timesteps = [], [], [], [], []
    for player in players:
        color = colors[player]
        player_states, player_actions, player_returns_to_go, player_attention_mask = format_tensors(
            move_indices[color],
            board_tensors[color],
            end_rewards[color],
            window_size,
            device,
            window_start=window_start,
            position_evaluations=position_evaluations[color] if shaping_rewards else None,
            shaping_rewards=shaping_rewards,
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
    states = torch.cat(states, dim=2).reshape(1, n_players * window_size, STATE_DIM)
    actions = torch.cat(actions, dim=2).reshape(1, n_players * window_size, ACT_DIM)
    returns_to_go = torch.cat(returns_to_go, dim=2).reshape(1, n_players * window_size, 1)
    attention_mask = torch.cat(attention_mask, dim=2).reshape(1, n_players * window_size)
    timesteps = torch.cat(timesteps, dim=2).reshape(1, n_players * window_size)

    input_dict = {
        "states": states,
        "actions": actions,
        "returns_to_go": returns_to_go,
        "attention_mask": attention_mask,
        "timesteps": timesteps,
    }
    if return_labels:
        input_dict["labels"] = actions
    return input_dict
