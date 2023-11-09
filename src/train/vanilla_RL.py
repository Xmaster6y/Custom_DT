"""
Model training using off-policy policy gradient algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import TrainingArguments
from collections import namedtuple, deque
from itertools import count
import random
import math

import src.utils.translate as translate
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.utils.dataset import TwoPlayersChessDataset
from src.utils.trainer import DecisionTransformerTrainer, compute_metrics

import sys
import os
import pathlib
cwd = os.getcwd()
sys.path.append(cwd)
import chess
import chess.engine
stockfish_root = list(pathlib.Path(f"{cwd}/stockfish-source/stockfish/").glob('*.exe'))[0]
engine = chess.engine.SimpleEngine.popen_uci(stockfish_root)

## engine.play(board, chess.engine.Limit(depth=4))

# TODO: Use argparse to set these variables

# Meta
DEBUG = True
TRAINING = False

# Config
STATE_DIM = 64
ACT_DIM = 4672
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Training
NAME = "first"
OVERWRITE = False
N_EPOCHS = 1
LOGGING_STEPS_RATIO = 0.01
EVAL_STEPS_RATIO = 0.1
if DEBUG:
    OUTPUT_DIR = "weights/debug"
    LOGGING_DIR = "logging/debug"
else:
    OUTPUT_DIR = f"weights/{NAME}"
    LOGGING_DIR = f"logging/{NAME}"
TRAIN_BATCH_SIZE = 50
GRADIENT_ACCUMULATION_STEPS = 1
EVAL_BATCH_SIZE = 500
LR = 1e-5


eval_generator = torch.Generator(device=DEVICE)
eval_generator.manual_seed(SEED)
eval_dataset = TwoPlayersChessDataset(
    file_name="data/chess_games_base/test_stockfish_5000.jsonl",
    board_to_tensor=translate.board_to_64tensor,
    act_dim=4672,
    state_dim=64,
    discount=0.99,
    window_size=10,
    generator=eval_generator,
    return_ids=True,
    eval_mode=True,
)
eval_dataset_len = len(eval_dataset)

train_generator = torch.Generator(device=DEVICE)
train_generator.manual_seed(SEED)
if DEBUG:
    train_dataset_file = "data/chess_games_base/test_stockfish_5000.jsonl"
else:
    train_dataset_file = "data/chess_games_base/train_stockfish_262k.jsonl"
train_dataset = TwoPlayersChessDataset(
    file_name=train_dataset_file,
    board_to_tensor=translate.board_to_64tensor,
    act_dim=4672,
    state_dim=64,
    discount=0.99,
    window_size=10,
    generator=train_generator,
    return_ids=True,
)
train_dataset_len = len(train_dataset)

conf = DecisionTransformerConfig(
    state_dim=STATE_DIM,
    act_dim=ACT_DIM,
)
model = DecisionTransformerModel(conf)

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """ Save a transition """
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, STATE_DIM, ACT_DIM):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(STATE_DIM, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, ACT_DIM)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

policy_net = DecisionTransformerModel(conf).to(DEVICE)
target_net = DecisionTransformerModel(conf).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict) # target net copies parameters of policy net

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True) #optimizer defaults to AdamW in Trainer
memory = ReplayMemory(1000)

steps_done = 0

# sometimes use model for choosing action, and sometimes we'll just sample one uniformly according to current value of epsilon
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
    


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return #abandonship I think
    transitions = memory.sample(BATCH_SIZE)
    # transpose the batch. this converts batch-array of transitions to transition of batch-arrays 
    # - stack overflow about it on the tutorial
    batch = Transition(*zip(*transitions))

    # compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which the simulation ended)
    # a mask of T/F with T being a state to be predicted on and 
    # F being a final state not to be predicted on
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net  
    # - Q(s,a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    # r + gamma x armax_a[Q(s', pi(s'))]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# At the beginning we reset the environment and obtain the initial state Tensor. 
# Then, we sample an action, execute it, observe the next state and the reward (always 1),
# and optimize our model once. When the episode ends (our model fails), we restart the loop.

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=DEVICE)
        done = terminated or truncated

        if terminated:
            next_state  = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsueeze(0)

        # store the transition in memory
        memory.push(state, action, next_state, reward)

        # move to the next_state
        state = next_state

        # perform one step of the optimization on the policy network
        optimize_model()

        # soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
        if done:
            break

print('Complete')










args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOGGING_DIR,
    overwrite_output_dir=DEBUG or OVERWRITE,
    logging_strategy="steps",
    logging_steps=int(LOGGING_STEPS_RATIO * train_dataset_len / TRAIN_BATCH_SIZE),
    prediction_loss_only=False,
    evaluation_strategy="steps",
    eval_steps=int(EVAL_STEPS_RATIO * train_dataset_len / TRAIN_BATCH_SIZE),
    save_strategy="steps",
    save_steps=int(EVAL_STEPS_RATIO * train_dataset_len / TRAIN_BATCH_SIZE),
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=N_EPOCHS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    run_name="latest",
)

trainer = DecisionTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
if TRAINING:
    trainer.train()
else:
    evaluation = trainer.evaluate()
    print(evaluation)
