"""
Model training using self-supervised learning.
"""

import torch
from transformers import TrainingArguments

import src.utils.translate as translate
from src.models.decision_transformer import DecisionTransformerConfig, DecisionTransformerModel
from src.utils.dataset import ChessDataset
from src.utils.trainer import DecisionTransformerTrainer, compute_metrics

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
eval_dataset = ChessDataset(
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
train_dataset = ChessDataset(
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
