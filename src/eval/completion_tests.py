"""
Simple completion test class.
"""


import os
import pathlib
import sys

import chess
import chess.engine
import jsonlines
import torch

import src.utils.translate as translate
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel

cwd = os.getcwd()
sys.path.append(cwd)

torch.set_printoptions(sci_mode=False)


class CompletionTest:
    def __init__(
        self,
        file_name: str,
        n_test_games: int,
        state_dim: int = 64,
        action_dim: int = 4672,
        discount: float = 1.0,
        shaping: bool = False,
        shaping_eval_depth: int = 8,
        single_player: bool = True,
        device: str = "cpu",
    ):
        self.file_name = file_name
        self.n_test_games = n_test_games
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.shaping = shaping
        self.shaping_eval_depth = shaping_eval_depth
        self.single_player = single_player
        self.board_to_tensor = translate.board_to_64tensor if state_dim == 64 else translate.board_to_64x12tensor
        if device == "cuda":
            if not torch.cuda.is_available():
                print("Cuda not available, running CompletionTest on cpu")
                device = "cpu"
        self.device = torch.device(device)
        self.conf = DecisionTransformerConfig(
            state_dim=STATE_DIM,
            act_dim=ACT_DIM,
        )
        self.model = DecisionTransformerModel(self.conf)
        self.model.to(self.device)
        stockfish_root = list(pathlib.Path(f"{cwd}/stockfish-source/stockfish/").glob("*.exe"))[0]
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_root)

    def prepare_data(self):
        sequences = []
        with jsonlines.open(self.file_name) as reader:
            i = 0
            for obj in reader:
                sequences.append(obj["moves"])
                i += 1
                if i >= self.n_test_games:
                    break

        games_states, games_actions, games_target_returns, games_seq_len = [], [], [], []
        # for seq in sequences:

        if self.single_player:
            states, actions, target_returns, seq_len = self.single_player_prep(sequences[5])
        else:
            states, actions, target_returns, seq_len = self.two_player_prep(sequences[5])

        games_states.append(states)
        games_actions.append(actions)
        games_target_returns.append(target_returns)
        games_seq_len.append(seq_len)

        return games_states, games_actions, games_target_returns, games_seq_len

    def single_player_prep(self, seq):
        pass

    def two_player_prep(self, seq):
        move_indices, board_tensors, end_rewards, boards = translate.encode_seq(seq, self.board_to_tensor, self.shaping)
        seq_len = len(board_tensors)

        action_seq = torch.nn.functional.one_hot(torch.tensor(move_indices, dtype=int), num_classes=self.action_dim)
        actions = action_seq.reshape(1, seq_len, self.action_dim).to(device=self.device, dtype=torch.float32)

        state_seq = torch.stack(board_tensors)
        states = state_seq.reshape(1, seq_len, self.state_dim).to(device=self.device, dtype=torch.float32)

        black_seq_len = seq_len // 2
        white_seq_len = seq_len - black_seq_len

        if self.shaping:
            white_evals = []
            black_evals = []
            for b_idx, b in enumerate(boards):
                info = self.engine.analyse(
                    b, chess.engine.Limit(depth=self.shaping_eval_depth), info=chess.engine.Info.SCORE
                )
                if b_idx % 2 == 0:
                    white_evals.append(info["score"].white().score(mate_score=10000) / 10000)
                else:
                    black_evals.append(info["score"].black().score(mate_score=10000) / 10000)
            white_evals = torch.tensor(white_evals, dtype=torch.float32, device=self.device)
            black_evals = torch.tensor(black_evals, dtype=torch.float32, device=self.device)
            white_result = torch.full((len(white_evals),), end_rewards[0], device=self.device)
            black_result = torch.full((len(black_evals),), end_rewards[1], device=self.device)

            white_returns = torch.sub(white_result, white_evals)
            black_returns = torch.sub(black_result, black_evals)

        else:
            black_returns = self.discount ** torch.arange(black_seq_len, device=self.device) * end_rewards[1]
            white_returns = self.discount ** torch.arange(white_seq_len, device=self.device) * end_rewards[0]

        condition = torch.arange(seq_len, device=self.device) % 2 == 0
        target_returns = torch.zeros(1, seq_len, 1, device=self.device, dtype=torch.float32)
        target_returns[:, condition, :] = white_returns.reshape(1, white_seq_len, 1)
        target_returns[:, ~condition, :] = black_returns.reshape(1, black_seq_len, 1)

        return states, actions, target_returns, seq_len

    def run_test(self):
        games_states, games_actions, games_target_returns, games_seq_len = self.prepare_data()

        for game_idx in range(self.n_test_games):
            states = games_states[game_idx]
            actions = games_actions[game_idx]
            target_returns = games_target_returns[game_idx]
            seq_len = games_seq_len[game_idx]

            timesteps = torch.arange(seq_len, device=DEVICE).reshape(1, seq_len)
            # forward pass
            with torch.no_grad():
                state_preds, action_preds, return_preds = self.model(
                    states=states,
                    actions=actions,
                    returns_to_go=target_returns,
                    timesteps=timesteps,
                    return_dict=False,
                )
            print(f"Game {game_idx + 1} -")
            print(f"State Predictions shape: {state_preds.shape}")
            print(f"Action Predictions shape: {action_preds.shape}")
            print(f"Return Predictions shape: {return_preds.shape}")
            print(f"Return Predictions: {return_preds}")


if __name__ == "__main__":
    FILE_NAME = "data/chess_games_base/test_stockfish_5000.jsonl"
    N = 10
    STATE_DIM = 64
    ACT_DIM = 4672
    DISCOUNT = 0.99
    SHAPING = True
    SINGLE_PLAYER = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CT = CompletionTest(
        file_name=FILE_NAME,
        n_test_games=N,
        state_dim=STATE_DIM,
        action_dim=ACT_DIM,
        discount=DISCOUNT,
        shaping=SHAPING,
        single_player=SINGLE_PLAYER,
        device=DEVICE,
    )
    CT.prepare_data()
    # CT.run_test()
