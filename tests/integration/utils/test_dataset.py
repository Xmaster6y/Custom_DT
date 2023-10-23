"""
Test dataset.
"""

import pathlib

import torch
from torch.utils.data import DataLoader

import src.utils.translate as translate
from src.utils.dataset import ChessDataset, custom_collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIRECTORY = pathlib.Path(__file__).parent.absolute()


class TestChessDataset:
    def test_dataset_loading(self, default_64_chess_dataset):
        assert len(default_64_chess_dataset) == 10
        assert isinstance(default_64_chess_dataset[0], dict)

    def test_dataset_batching(self, default_64_chess_dataset):
        dataloader = DataLoader(
            dataset=default_64_chess_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )
        for batch in dataloader:
            assert batch["actions"].shape == (2, 10, 4672)
            assert batch["states"].shape == (2, 10, 64)
            assert batch["returns_to_go"].shape == (2, 10, 1)
            break

    def test_dataset_random_generator(self):
        generator = torch.Generator()
        generator.manual_seed(42)
        dataset = ChessDataset(
            file_name=f"{DIRECTORY}/../assets/test_stockfish_10.jsonl",
            board_to_tensor=translate.board_to_64tensor,
            act_dim=4672,
            state_dim=64,
            discount=0.99,
            window_size=10,
            generator=generator,
            return_ids=True,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=custom_collate_fn,
            generator=generator,
        )
        for batch in dataloader:
            assert batch["gameid"][0] == "8b9722ba-feb8-11ec-b8c1-0b557a78fa69"
            assert batch["timesteps"][0, 0] == 5
            break
        for batch in dataloader:
            assert batch["gameid"][0] == "874c0e6e-feb8-11ec-b8c1-0b557a78fa69"
            assert batch["timesteps"][0, 0].item() == 53
            break
