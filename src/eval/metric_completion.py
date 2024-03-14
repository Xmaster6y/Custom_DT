"""
(Deprecated) Tests dataset formatting.

With this class, a user can test that the chess dataset augmented with
Stockfish evaluation shaping rewards is formatted correctly for the one and two
player settings. This is confirmed by printing out the game dictionary and
doing a forward pass through a Decision Transformer. Uses deprecated encodings.

Typical usage example:

```python
    FILE_NAME = "data/chess_games_base/test_stockfish_5000.jsonl"
    N = 3
    SEED = 42
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)

    CT = StockfishEvalTest(
        file_name=FILE_NAME,
        n_test_games=N,
        generator=generator,
    )
    CT.run_test()
```
"""

import torch

import src.utils.translate as translate
from src.metric.stockfish import StockfishMetric
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.utils.dataset import OnePlayerChessDataset, TwoPlayersChessDataset

torch.set_printoptions(sci_mode=False)


class StockfishEvalTest:
    def __init__(
        self,
        file_name: str,
        n_test_games: int,
        generator: torch.Generator,
        state_dim: int = 64,
        action_dim: int = 4672,
        window_size: int = 10,
        shaping_rewards: bool = False,
        one_player: bool = False,
        device: str = "cpu",
        default_platform: str = "auto",
    ):
        self.file_name = file_name
        self.n_test_games = n_test_games
        self.generator = generator
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.shaping_rewards = shaping_rewards
        self.one_player = one_player
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

        self.stockfish_metric = StockfishMetric(default_platform=default_platform)

    def run_test(self):
        data_config = {
            "file_name": self.file_name,
            "board_to_tensor": self.board_to_tensor,
            "act_dim": self.action_dim,
            "state_dim": self.state_dim,
            "window_size": self.window_size,
            "generator": self.generator,
            "return_ids": True,
            "eval_mode": True,
            "shaping_rewards": self.shaping_rewards,
            "stockfish_metric": self.stockfish_metric,
        }
        if self.one_player:
            eval_dataset = OnePlayerChessDataset(**data_config)
        else:
            eval_dataset = TwoPlayersChessDataset(**data_config)

        for game_idx in range(self.n_test_games):
            game = eval_dataset[game_idx]
            print(f"Game {game_idx}: ID: {game['gameid']}")
            print(f"Shaping rewards: {self.shaping_rewards}, One player: {self.one_player}")
            print(game)
            print("-" * 40)
        self.stockfish_metric.engine.quit()

        with torch.no_grad():
            _, _, _ = self.model(
                states=game["states"].unsqueeze(0),
                actions=game["actions"].unsqueeze(0),
                returns_to_go=game["returns_to_go"].unsqueeze(0),
                timesteps=game["timesteps"],
                return_dict=False,
            )


if __name__ == "__main__":
    # Static Config
    FILE_NAME = "data/chess_games_base/test_stockfish_5000.jsonl"
    N = 3
    STATE_DIM = 64
    ACT_DIM = 4672
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    WINDOW_SIZE = 10
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)

    # runs each combo of shaping rewards and # of players
    for i in range(4):
        CT = StockfishEvalTest(
            file_name=FILE_NAME,
            n_test_games=N,
            generator=generator,
            state_dim=STATE_DIM,
            action_dim=ACT_DIM,
            window_size=WINDOW_SIZE,
            shaping_rewards=bool(i % 2),
            one_player=bool(i // 2),
            device=DEVICE,
        )
        CT.run_test()
