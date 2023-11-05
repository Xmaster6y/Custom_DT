"""
HuggingFace-like Trainer class for training and evaluating models.
"""

import evaluate
import numpy as np
from transformers import Trainer


class DecisionTransformerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        preds = model(**inputs)
        sq_diff = (preds["action_preds"] - inputs["actions"]) ** 2
        return (sq_diff.mean(), preds) if return_outputs else sq_diff.mean()


def compute_metrics(eval_preds):
    metrics = ["accuracy"]
    avg_metrics = ["f1", "precision", "recall"]
    outputs, labels = eval_preds
    state_preds, action_preds, return_preds, last_hidden_states = outputs
    predictions = np.argmax(action_preds, axis=-1)
    class_labels = np.argmax(labels, axis=-1)
    metric_dict = {}
    for metric_name in metrics:
        metric = evaluate.load(metric_name)
        metric_dict.update(metric.compute(predictions=predictions.flatten(), references=class_labels.flatten()))
    for metric_name in avg_metrics:
        metric = evaluate.load(metric_name)
        metric_dict.update(
            metric.compute(predictions=predictions.flatten(), references=class_labels.flatten(), average="micro")
        )
    sq_diff = (action_preds - labels) ** 2
    metric_dict["loss_mse"] = sq_diff.mean()
    return metric_dict
