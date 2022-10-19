"""Methods for evaluating model performance."""
import numpy as np
from datasets import load_metric


def compute_metrics(predictions,
                    labels,
                    metric=load_metric("seqeval"),
                    binary_evaluation=False) -> dict:
    """Returns a dictionary of operating point statistics (Precision/Recall/F1)."""

    if binary_evaluation:
        # convert all labels to PHI or not PHI
        predictions = [['PHI' if w != 'O' else 'O' for w in sequence]
                       for sequence in predictions]
        labels = [['PHI' if w != 'O' else 'O' for w in sequence]
                  for sequence in labels]

    # convert to BIOC
    # true_predictions = convert_to_bio_scheme(true_predictions)
    # true_labels = convert_to_bio_scheme(true_labels)

    return metric.compute(predictions=predictions, references=labels)
