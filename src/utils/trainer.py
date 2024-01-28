"""
HuggingFace-like Trainer class for training and evaluating models.
"""

from typing import Dict, Tuple, Union

import evaluate
import numpy as np
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction

from src.models.decision_transformer import DecisionTransformerModel


class DecisionTransformerTrainer(Trainer):
    """
    Extends Trainer from transformers to compute MSE loss.
    """

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
        sq_diff = (preds["action_preds"] - inputs["actions"]) ** 2
        return (sq_diff.mean(), preds) if return_outputs else sq_diff.mean()


def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Computes metrics for a batch of predictions.

    Args:
        eval_preds: batch of predictions from DecisionTransformerModel

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
