"""
Simple completion test.
"""

import chess
import jsonlines
import torch

import src.utils.translate as translate
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel

FILE_NAME = "data/chess_games_base/test_stockfish_5000.jsonl"
N = 10
STATE_DIM = 64
ACT_DIM = 4672
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_RETURN = 1.0
BOARD_TO_TENSOR = translate.board_to_64tensor if STATE_DIM == 64 else translate.board_to_64x12tensor

sequences = []

with jsonlines.open(FILE_NAME) as reader:
    i = 0
    for obj in reader:
        sequences.append(obj["moves"])
        i += 1
        if i >= N:
            break

conf = DecisionTransformerConfig(
    state_dim=STATE_DIM,
    act_dim=ACT_DIM,
)
model = DecisionTransformerModel(conf)
model.to(DEVICE)

move_indices, board_tensors, end_rewards = translate.encode_seq(sequences[3], BOARD_TO_TENSOR)
seq_len = len(board_tensors)

action_seq = torch.nn.functional.one_hot(torch.tensor(move_indices, dtype=int), num_classes=ACT_DIM)
action_seq = torch.concat([action_seq, torch.zeros(1, ACT_DIM)], dim=0)
actions = action_seq.reshape(1, seq_len, ACT_DIM).to(device=DEVICE, dtype=torch.float32)

state_seq = torch.stack(board_tensors)
states = state_seq.reshape(1, seq_len, STATE_DIM).to(device=DEVICE, dtype=torch.float32)

board = chess.Board()
state = BOARD_TO_TENSOR(board)

rewards = torch.zeros(1, seq_len, device=DEVICE, dtype=torch.float32)
if seq_len > 3:
    rewards[0, -3] = end_rewards[0]
    rewards[0, -2] = end_rewards[1]

target_return = torch.full((1, seq_len, 1), TARGET_RETURN, device=DEVICE, dtype=torch.float32)
timesteps = torch.arange(seq_len, device=DEVICE).reshape(1, seq_len)
attention_mask = torch.ones(1, seq_len, device=DEVICE, dtype=torch.float32)

# forward pass
with torch.no_grad():
    state_preds, action_preds, return_preds = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=target_return,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )
print(state_preds.shape)
print(action_preds.shape)
print(return_preds.shape)
print(return_preds)
