"""
Provides the Stockfish evaluations of board positions.

Typical usage example:

```python
>>> sm = StockfishMetric()
>>> chess_board = chess.Board()
>>> sm.eval_board(chess_board)

>>> sm = StockfishMetric()
>>> seq = "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"
>>> sm.eval_sequence(sequence)
```
"""

import pathlib
import sys
from typing import List, Tuple, Union

import chess
import chess.engine

__ALL__ = ["stockfish_eval_board", "stockfish_eval_sequence"]


class StockfishMetric:
    """
    Provides the Stockfish evaluations of board positions.

    Attributes:
        default_platform: the system platform on which the Stockfish engine is run.
        engine: the stockfish engine used in evaluations.
    """

    def __init__(self, default_platform: str = "auto"):
        """
        Initializes the StockfishMetric class.

        Args:
            default_platform: the system platform on which the Stockfish engine is run.

        Raises:
            ValueError: an error occurred if the platform is not recognized.
        """
        if default_platform == "auto":
            if sys.platform in ["linux"]:
                platform = "linux"
            elif sys.platform in ["win32", "cygwin"]:
                platform = "windows"
            elif sys.platform in ["darwin"]:
                platform = "macos"
            else:
                raise ValueError(f"Unknown platform {sys.platform}")
        else:
            platform = default_platform

        if platform in ["linux", "macos"]:
            exec_re = "stockfish*"
        elif platform == "windows":
            exec_re = "stockfish*.exe"
        else:
            raise ValueError(f"Unknown platform {platform}")

        stockfish_root = list(pathlib.Path("stockfish-source/stockfish/").glob(exec_re))[0]
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_root)

    def _validate_params(
        self,
        parameter_name: str,
        input: object,
        expected_type: type,
        permitted_values: list = None,
    ):
        """
        Validates the type of the parameter and whether the value is in the permitted values list.

        Args:
            input: input parameter to be validated
            expected_type: expected type of the input parameter
            permitted_values: list of permitted values for the input parameter.
                If None, then no permitted values are checked.

        Raises:
            TypeError: an error occurred if the input parameter is not of the expected type.
            ValueError: an error occurred if the input parameter is not in the permitted values list.
        """
        if not isinstance(input, expected_type):
            raise TypeError(f"Invalid type for parameter '{parameter_name}'. Expected type: {str(expected_type)}")
        if permitted_values is not None:
            if input not in permitted_values:
                raise ValueError(
                    f"Invalid value for parameter '{parameter_name}'. Permitted values: {str(permitted_values)}"
                )

    def eval_board(
        self,
        board: chess.Board,
        player: str = "white",
        evaluation_depth: int = 8,
        return_info: bool = False,
    ) -> Union[float, Tuple[float, dict]]:
        """
        Provides the Stockfish evaluation of a board position.

        Args:
            board: board position to be evaluated
            player: player for which the evaluation is performed. If 'both' then the board evaluations will
                alternate perspectives, with the white perspective being first.
            evaluation_depth: depth of search for Stockfish engine. Increasing the depth of the search increases
                runtime.
            return_info: if True, then the function will return the Stockfish info object

        Returns:
            Stockfish evaluation of the board position and optionally the Stockfish info object

        Typical usage example:
        ```python
        >>> sm = StockfishMetric()
        >>> chess_board = chess.Board()
        >>> sm.eval_board(chess_board)
        ```
        """
        # input validation
        self._validate_params("player", player, str, ["white", "black", "both"])
        self._validate_params("evaluation_depth", evaluation_depth, int, list(range(1, 13)))
        ########################################
        info = self.engine.analyse(board, chess.engine.Limit(depth=evaluation_depth))
        if player == "white" or (
            player == "both" and board.turn == chess.BLACK
        ):  # white has just moved if turn is black
            evaluation = info["score"].white().score(mate_score=10000) / 10000
        else:
            evaluation = info["score"].black().score(mate_score=10000) / 10000
        return (evaluation, info) if return_info else evaluation

    def eval_sequence(self, sequence: str, player: str = "white", evaluation_depth: int = 8) -> List[dict]:
        """
        Provides the Stockfish evaluation of board position for a list of boards.

        Args:
            sequence: sequence of moves in standard chess notation.
            player: player for which the evaluation is performed. If 'both' then the board evaluations will
                alternate perspectives, with the white perspective being first.
            evaluation_depth: Depth of search for Stockfish engine. Increasing the depth of the search increases
                runtime.

        Returns:
            a list of Stockfish evaluations for each board position

        Typical usage example:
        ```python
        >>> sm = StockfishMetric()
        >>> seq = "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"
        >>> sm.eval_sequence(sequence)
        ```
        """
        # input validation
        self._validate_params("sequence", sequence, str, None)
        self._validate_params("player", player, str, ["white", "black", "both"])
        self._validate_params("evaluation_depth", evaluation_depth, int, list(range(1, 13)))
        ########################################

        evaluations = []
        board = chess.Board()
        for alg_move in sequence.split():
            if alg_move.endswith("."):
                continue
            board.push_san(alg_move)
            evaluation = self.eval_board(board, player, evaluation_depth)
            evaluations.append(evaluation)
        return evaluations
