"""
Deprecated.
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
DISCOUNT = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

move_indices, board_tensors, end_rewards, _ = translate.encode_seq(sequences[3], BOARD_TO_TENSOR)
seq_len = len(board_tensors)

action_seq = torch.nn.functional.one_hot(torch.tensor(move_indices, dtype=int), num_classes=ACT_DIM)
actions = action_seq.reshape(1, seq_len, ACT_DIM).to(device=DEVICE, dtype=torch.float32)

state_seq = torch.stack(board_tensors)
states = state_seq.reshape(1, seq_len, STATE_DIM).to(device=DEVICE, dtype=torch.float32)

board = chess.Board()
state = BOARD_TO_TENSOR(board)

black_seq_len = seq_len // 2
white_seq_len = seq_len - black_seq_len
black_returns = DISCOUNT ** torch.arange(black_seq_len, device=DEVICE) * end_rewards[1]
white_returns = DISCOUNT ** torch.arange(white_seq_len, device=DEVICE) * end_rewards[0]

condition = torch.arange(seq_len, device=DEVICE) % 2 == 0
target_returns = torch.zeros(1, seq_len, 1, device=DEVICE, dtype=torch.float32)
target_returns[:, condition, :] = white_returns.reshape(1, white_seq_len, 1)
target_returns[:, ~condition, :] = black_returns.reshape(1, black_seq_len, 1)

timesteps = torch.arange(seq_len, device=DEVICE).reshape(1, seq_len)
attention_mask = torch.ones(1, seq_len, device=DEVICE, dtype=torch.float32)

print(timesteps.shape)
print(states.shape)
print(states)
print(actions.shape)
print(actions)
print(target_returns.shape)
print(target_returns)

# # forward pass
# with torch.no_grad():
#     state_preds, action_preds, return_preds = model(
#         states=states,
#         actions=actions,
#         returns_to_go=target_returns,
#         timesteps=timesteps,
#         return_dict=False,
#     )
# print(state_preds.shape)
# print(action_preds.shape)
# print(return_preds.shape)
# print(return_preds)
