"""
HuggingFace-like Trainer class for training and evaluating models.
"""

import evaluate
import numpy as np
from transformers import Trainer


class DecisionTransformerTrainer(Trainer):
    def compute_loss(self, model, inputs):
        preds = model(**inputs)
        sq_diff = (preds["action_preds"] - inputs["actions"]) ** 2
        return sq_diff.mean()


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metric_dict = metric.compute(predictions=predictions, references=labels)
    sq_diff = (predictions - labels) ** 2
    metric_dict["loss_mse"] = sq_diff.mean()
    return metric_dict
