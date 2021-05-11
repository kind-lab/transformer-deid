import importlib
from typing import List, Optional, Union
import numpy as np

from sklearn.metrics import accuracy_score

import datasets

from sklearn.metrics import accuracy_score
from sklearn.exceptions import UndefinedMetricWarning


import datasets

_DESCRIPTION = """
Produces operating point statistics for an element-wise comparison
of two sets of labels. Does *not* merge across entities.
"""

_CITATION = """N/A."""

_KWARGS_DESCRIPTION = """
Produces operating point statistics for an element-wise comparison
of two sets of labels. Does *not* merge across entities.

Args:
    predictions: List of List of predicted labels
    references: List of List of reference labels
    sample_weight: Array-like of shape (n_samples,), weights for individual samples. default: None
    zero_division: A value to substitute for the statistic when encountering a divide by zero exception.
        Allowed values include 0, 1, "warn". "warn" acts as 0 and raises a warning.

Returns:
    'scores': dict. Summary of the scores for overall and per type
        Overall:
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure,
        Per type:
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure
Examples:

    >>> predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> seqeval = datasets.load_metric("seqeval")
    >>> results = seqeval.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['MISC', 'PER', 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']
    >>> print(results["overall_f1"])
    0.5
    >>> print(results["PER"]["f1"])
    1.0

See also:
    seqeval library, which this class aims to emulate.
    huggingface datasets library, which this sub-classes from.
"""

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TokenEvaluation(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/alistairewj/transformers-deid",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value('string')),
                'references': datasets.Sequence(datasets.Value('string')),
            }),
            codebase_urls=["https://github.com/alistairewj/transformers-deid"],
            reference_urls=["https://github.com/alistairewj/transformers-deid"]
        )

    def _safely_return_value(zero_division):
        if zero_division != 'warn':
            return zero_division
        else:
            warnings.warn('Division by zero.', UndefinedMetricWarning, stacklevel=2)
            return 0

    def _precision(true_positive, false_positive, zero_division: Union[str, int]='warn'):
        denom = np.sum(true_positive) + np.sum(false_positive)
        if denom == 0:
            return _safely_return_value(zero_division)
        return np.sum(true_positive) / denom

    def _recall(true_positive, positive, zero_division: Union[str, int]='warn'):
        denom = np.sum(positive)
        if denom == 0:
            return _safely_return_value(zero_division)
        return np.sum(true_positive) / denom

    def _f1(precision, recall, zero_division: Union[str, int]='warn'):
        denom = precision + recall
        if denom == 0:
            return _safely_return_value(zero_division)
        return 2*precision*recall / denom

    def _compute(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
        sample_weight: Optional[List[int]] = None,
        zero_division: Union[str, int] = "warn",
    ):
        # extract all possible entities
        predicted_entities = set([entity for entities in predictions for entity in entities])
        reference_entities = set([entity for entities in predictions for entity in entities])

        labels = sorted(predicted_entities.union(reference_entities))

        # compute per-class scores.
        scores = {}

        all_pos, all_tp, all_fp = [], [], []
        for label in labels:
            label_pos, label_tp, label_fp = [], [], []
            for pred, target in zip(predictions, references):
                # convert to numpy for array calculations
                pred = np.asarray(pred)
                target = np.asarray(target)

                idxTrue = target == label
                idxPredPos = pred == label

                pos = idxTrue.sum()
                # pred_pos = idxPredPos.sum()
                tp = np.sum(idxTrue & idxPredPos)
                fp = np.sum(idxPredPos & ~idxTrue)


                label_pos.append(pos)
                label_tp.append(tp)
                label_fp.append(fp)
            
            prec = _precision(label_tp, label_fp)
            recall =  _recall(label_tp, label_pos)
            f1 = _f1(prec, recall)

            # upweight samples if requested
            if sample_weight is not None:
                label_tp = [p*w for p, w in zip(label_tp, sample_weight)]
                label_fp = [p*w for p, w in zip(label_fp, sample_weight)]
                label_pos = [p*w for p, w in zip(label_pos, sample_weight)]

            scores[label] = {
                "precision": prec,
                "recall": recall,
                "f1": f1,
                "number": np.sum(label_pos)
            }

            # retain the individual counts for micro average
            all_pos.extend(label_pos)
            all_tp.extend(label_tp)
            all_fp.extend(label_fp)
        

        
        # micro average
        prec = _precision(all_tp, all_fp)
        recall =  _recall(all_tp, all_pos)
        f1 = _f1(prec, recall)
        scores["overall_precision"] = prec
        scores["overall_recall"] = recall
        scores["overall_f1"] = f1

        ref_unravel = [entity for entities in references for entity in entities]
        pred_unravel = [entity for entities in predictions for entity in entities]
        scores["overall_accuracy"] = accuracy_score(y_true=ref_unravel, y_pred=pred_unravel, sample_weight=sample_weight)

        return scores
