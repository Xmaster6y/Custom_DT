"""
HuggingFace-like Trainer class for training and evaluating models.
"""

from typing import Dict, Tuple, Union

import evaluate
import numpy as np
import torch
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction

from src.models.decision_transformer import DecisionTransformerModel


class DecisionTransformerTrainer(Trainer):
    """
    Extends Trainer from transformers to compute MSE loss.
    """

    def __init__(self, *args, **kwargs):
        self.pad_token_id = kwargs.pop("pad_token_id")
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model: DecisionTransformerModel, inputs: dict, return_outputs=False
    ) -> Union[Tuple[float, dict], float]:
        """
        Computes MSE loss for a model on a batch of inputs.

        Args:
            model: model to compute loss for
            return_outputs: whether to return the model outputs
            inputs: batch of inputs

        Returns:
            MSE loss for the model on the batch of inputs
        """
        preds = model(**inputs)
        labels = inputs["actions"]
        # creating label mask to ignore padding tokens
        label_maxes = torch.max(labels, dim=-1).indices != self.pad_token_id
        label_mask = label_maxes.unsqueeze(-1).expand_as(labels)
        sq_diff = torch.where(label_mask, (preds["action_preds"] - inputs["actions"]) ** 2, float("nan"))
        return (sq_diff.nanmean(), preds) if return_outputs else sq_diff.mean()


def compute_metrics(eval_preds: EvalPrediction, pad_token_id=0) -> Dict[str, float]:
    """
    Computes metrics for a batch of predictions.

    Args:
        eval_preds: batch of predictions from DecisionTransformerModel
        pad_token_id: id of the padding token

    Returns:
        dictionary of metrics - accuracy, f1, precision, recall, and MSE loss

    Typical usage example:
    ```python
        trainer = DecisionTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    ```
    """
    metrics = ["accuracy"]
    avg_metrics = ["f1", "precision", "recall"]
    outputs, labels = eval_preds
    state_preds, action_preds, return_preds, last_hidden_states = outputs
    predictions = np.argmax(action_preds, axis=-1).flatten()
    class_labels = np.argmax(labels, axis=-1).flatten()
    predictions = predictions[class_labels != pad_token_id]
    class_labels = class_labels[class_labels != pad_token_id]
    metric_dict = {}
    for metric_name in metrics:
        metric = evaluate.load(metric_name)
        metric_dict.update(metric.compute(predictions=predictions, references=class_labels))
    for metric_name in avg_metrics:
        metric = evaluate.load(metric_name)
        metric_dict.update(metric.compute(predictions=predictions, references=class_labels, average="micro"))
    sq_diff = (action_preds - labels) ** 2
    metric_dict["loss_mse"] = sq_diff.mean()
    return metric_dict
