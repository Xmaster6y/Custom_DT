"""
Module implementing the Stockfish metric class.
"""

from typing import Tuple, Union

import chess
import chess.engine

__ALL__ = ["stockfish_eval_board", "stockfish_eval_sequence"]


def validate_params(parameter_name, input, expected_type, permitted_values):
    """
    This function will validate the type of the parameter and whether the value is in the permitted values list

    Args:

        input (any): Input parameter to be validated

        expected_type (type): Expected type of the input parameter

        permitted_values (list): List of permitted values for the input parameter.
            If None, then no permitted values are checked.

    Returns:

        None
    """
    if not isinstance(input, expected_type):
        raise TypeError(f"Invalid type for parameter '{parameter_name}'. Expected type: {str(expected_type)}")
    if permitted_values is not None:
        if input not in permitted_values:
            raise ValueError(
                f"Invalid value for parameter '{parameter_name}'. Permitted values: {str(permitted_values)}"
            )


def stockfish_eval_board(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    player: str = "white",
    evaluation_depth: int = 8,
    return_info: bool = False,
) -> Union[float, Tuple[float, dict]]:
    """
    This function will provide the Stockfish evaluation of a board position

    Args:

        board (chess.Board): Board position to be evaluated

        engine (chess.engine.SimpleEngine): Stockfish engine that performs the evaluation

        player (str): Player for which the evaluation is performed. If 'both' then the board evaluations will
            alternate perspectives, with the white perspective being first.
            - Options: ['white', 'black', 'both']
            - Default: 'white'

        evaluation_depth (int): Depth of search for Stockfish engine. Increasing the depth of the search increases
            runtime.

        return_info (bool): If True, then the function will return the Stockfish info object

    Returns:

            Stockfish evaluation of the board position and optionally the Stockfish info object
    """
    info = engine.analyse(board, chess.engine.Limit(depth=evaluation_depth))
    if player == "white" or (player == "both" and board.turn == chess.BLACK):  # white has just moved if turn is black
        evaluation = info["score"].white().score(mate_score=1000) / 1000
    else:
        evaluation = info["score"].black().score(mate_score=1000) / 1000
    return (evaluation, info) if return_info else evaluation


def stockfish_eval_sequence(
    sequence: str, engine: chess.engine.SimpleEngine, player: str = "white", evaluation_depth: int = 8
) -> list:
    """
    This function will provide the Stockfish evaluation of board position for a list of boards

    Args:

        sequence (str): Sequence of moves in standard chess notation.

        engine (chess.engine.SimpleEngine): Stockfish engine that performs the evaluation

        player (str): Player for which the evaluation is performed. If 'both' then the board evaluations will
            alternate perspectives, with the white perspective being first.
            - Options: ['white', 'black', 'both']
            - Default: 'white'

        evaluation_depth (int): Depth of search for Stockfish engine. Increasing the depth of the search increases
            runtime.
            - Options: [4...12]
            - Default: 8

    Returns:

        a list of Stockfish evaluations for each board position
    """
    # input validation
    validate_params("sequence", sequence, str, None)
    validate_params("engine", engine, chess.engine.SimpleEngine, None)
    validate_params("player", player, str, ["white", "black", "both"])
    validate_params("evaluation_depth", evaluation_depth, int, list(range(4, 13)))
    ########################################

    evaluations = []
    board = chess.Board()
    for alg_move in sequence.split():
        if alg_move.endswith("."):
            continue
        board.push_san(alg_move)
        evaluation = stockfish_eval_board(board, engine, player, evaluation_depth)
        evaluations.append(evaluation)
    return evaluations
