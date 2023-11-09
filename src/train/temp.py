import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import TrainingArguments
from collections import namedtuple, deque
from itertools import count
import random
import math


import sys
import os
import pathlib
cwd = os.getcwd()
sys.path.append(cwd)
import chess
import chess.engine
stockfish_root = list(pathlib.Path(f"{cwd}/stockfish-source/stockfish/").glob('*.exe'))[0]
engine = chess.engine.SimpleEngine.popen_uci(stockfish_root)

import src.utils.translate as translate
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.utils.dataset import TwoPlayersChessDataset
from src.utils.trainer import DecisionTransformerTrainer, compute_metrics

STATE_DIM = 64
ACT_DIM = 4672
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOARD_TO_TENSOR = translate.board_to_64tensor if STATE_DIM == 64 else translate.board_to_64x12tensor

simple_sequence = "1. d4 d5 2. c4 e6 3. e3 Nd7 4. cxd5 exd5 5. Nc3 Ngf6 6. h3 Bd6"

move_indices, board_tensors, end_rewards, boards = translate.encode_seq(simple_sequence, BOARD_TO_TENSOR, return_boards=True)

print("move indices")
print(move_indices)
print("board tensors")
print(board_tensors)
print("end rewards")
print(type(end_rewards))

# print(boards)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

conf = DecisionTransformerConfig(
    state_dim=STATE_DIM,
    act_dim=ACT_DIM,
)

policy_net = DecisionTransformerModel(conf).to(DEVICE)

steps_done = 0


def select_action(board, board_tensor):

    # stand-in action and return-to-go
    dummy_action_seq = torch.nn.functional.one_hot(torch.tensor([1], dtype=int), num_classes=ACT_DIM)
    dummy_action = dummy_action_seq.reshape(1, 1, ACT_DIM).to(device=DEVICE, dtype=torch.float32)
    dummy_return_to_go = torch.zeros((1, 1, 1), device=DEVICE, dtype=torch.float32)

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done/EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            _, action_preds, _ = policy_net(
                states=board_tensor.reshape(1,1,STATE_DIM).to(device=DEVICE, dtype=torch.float32), 
                actions=dummy_action, 
                returns_to_go=dummy_return_to_go,
                timesteps=torch.arange(1, device=DEVICE).reshape(1, 1),
                return_dict=False) #.max(1)[1].view(1,1)
            return action_preds.max(2)[1]
            # .max(1) will return the largest column value of each row.
            # second column on max result is index of where max elem was
            # found, so we pick acion with the larger expected reward
            # basically just picks left or right based on predicted expected value
    else:
        ### sample from action space in chess bot. not random but still adds move diversity I think
        move = engine.play(board, chess.engine.Limit(depth=1)).move
        promotion = move.promotion
        if promotion is not None and promotion != chess.QUEEN:  # Underpromotion
            direction = (move.to_square % 8) - (move.from_square % 8)
            extra_index = (promotion - 2) + 3 * (direction + 1) + 9 * move.from_square
            return 4096 + extra_index
        else:
            return torch.tensor([[move.from_square + 64 * move.to_square]], device=DEVICE, dtype=torch.long)



print(select_action(boards[0], board_tensors[0]))