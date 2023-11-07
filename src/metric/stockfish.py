"""
Module implementing the Stockfish metric class.
"""

import chess
import chess.engine

__ALL__ = ["stockfish_eval"]


def validate_params(parameter_name, input, expected_type, permitted_values):
    """
    This function will validate the type of the parameter and whether the value is in the permitted values list

    Args:

        input (any): Input parameter to be validated

        expected_type (type): Expected type of the input parameter

        permitted_values (list): List of permitted values for the input parameter. If None, then no permitted values are checked.

    Returns:

        None
    """
    if not isinstance(input, expected_type):
        raise TypeError(
            f"Invalid type for parameter '{parameter_name}'. Expected type: {str(expected_type)}"
        )
    if permitted_values:
        if input not in permitted_values:
            raise ValueError(
                f"Invalid value for parameter '{parameter_name}'. Permitted values: {str(permitted_values)}"
            )


def stockfish_eval(
    boards: list, engine: chess.engine.SimpleEngine, player: str = "white", evaluation_depth: int = 8
) -> list:
    """
    This function will provide the Stockfish evaluation of board position for a list of boards

    Args:

        boards (list): List of chess.Board objects

        engine (chess.engine.SimpleEngine): Stockfish engine that performs the evaluation

        player (str): Player for which the evaluation is performed. If 'both' then the board evaluations will alternate perspectives, with the white perspective being first. - Options: ['white', 'black', 'both'] - Default: 'white'

        evaluation_depth (int): Depth of search for Stockfish engine. Increasing the depth of the search increases runtime. - Options: [4...12] - Default: 8

    Returns:

        a list of Stockfish evaluations for each board position
    """
    # input validation
    validate_params("boards", boards, list, None)
    validate_params("engine", engine, chess.engine.SimpleEngine, None)
    validate_params("player", player, str, ["white", "black", "both"])
    validate_params("evaluation_depth", evaluation_depth, int, list(range(4, 13)))
    ########################################

    evaluations = []
    for idx, board in enumerate(boards):
        info = engine.analyse(board, chess.engine.Limit(depth=evaluation_depth))
        if (
            player != "black"
            and player == "both"
            and idx % 2 == 0
            or player != "black"
            and player != "both"
            and player == "white"
        ):
            evaluations.append(info["score"].white().score(mate_score=10000) / 10000)
        elif player != "black" and player == "both" or player == "black":
            evaluations.append(info["score"].black().score(mate_score=10000) / 10000)

    return evaluations
