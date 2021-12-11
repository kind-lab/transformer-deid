import numpy as np
import sys
import math
import pprint
from sklearn import metrics
from datasets import load_metric

sys.path.append("..")
from transformer_deid import evaluation

metric_dir = "../transformer_deid/token_evaluation.py"
metric = load_metric(metric_dir)

text_labels = ['O', 'AGE', 'CONTACT', 'DATE', 'ID', 'LOCATION', 'NAME', 'PROFESSION']


def test_class_metrics():
    """
    Generate 10 test pairs randomly, compare calculated metrics to sklearn.metrics
    """
    # TODO: Note that -100 values are not included in test cases
    predictions = np.random.randint(0, high=len(text_labels) - 1, size=(10, 250))  # TODO: Not sure how to handle 'O'
    references = np.random.randint(0, high=len(text_labels) - 1, size=(10, 250))
    predictions1d = predictions.flatten()
    references1d = references.flatten()

    predicted_entities = set([entity for entities in predictions for entity in entities])
    reference_entities = set([entity for entities in references for entity in entities])
    labels = sorted(predicted_entities.union(reference_entities))

    comp_metrics = evaluation.compute_metrics(predictions=predictions, labels=references, label_list=text_labels, metric=metric, binary_evaluation=False)
    target_metrics = metrics.classification_report(y_pred=predictions1d, y_true=references1d, output_dict=True, labels=list(range(8)), target_names=text_labels)

    for idx_label, text_label in enumerate(text_labels):
        if text_label == 'O' or idx_label not in labels:
            continue
        assert math.isclose(comp_metrics[text_label]['f1'], target_metrics[text_label]['f1-score'], rel_tol=1e-6)
        assert math.isclose(comp_metrics[text_label]['number'], target_metrics[text_label]['support'], rel_tol=1e-6)
        assert math.isclose(comp_metrics[text_label]['precision'], target_metrics[text_label]['precision'], rel_tol=1e-6)
        assert math.isclose(comp_metrics[text_label]['recall'], target_metrics[text_label]['recall'], rel_tol=1e-6)
    # TODO: validate overall accuracy, precision, recall, f1

def test_binary_metrics():
    """
    Generate 10 test pairs randomly, compare calculated binary metrics to sklearn.metrics
    """
    predictions = np.random.randint(0, high=len(text_labels) - 1, size=(10, 250))
    references = np.random.randint(1, high=len(text_labels) - 1, size=(10, 250))

    binary_predictions = predictions.flatten() == 0  # TODO: should be 6?
    binary_references = references.flatten() == 0

    comp_metrics = evaluation.compute_metrics(predictions=predictions, labels=references, label_list=text_labels, metric=metric, binary_evaluation=True)
    target_metrics = metrics.classification_report(y_pred=binary_predictions, y_true=binary_references, output_dict=True)

    assert math.isclose(comp_metrics['PHI']['f1'], target_metrics['False']['f1-score'], rel_tol=1e-6)
    assert math.isclose(comp_metrics['PHI']['number'], target_metrics['False']['support'], rel_tol=1e-6)
    assert math.isclose(comp_metrics['PHI']['precision'], target_metrics['False']['precision'], rel_tol=1e-6)
    assert math.isclose(comp_metrics['PHI']['recall'], target_metrics['False']['recall'], rel_tol=1e-6)
    assert math.isclose(comp_metrics['overall_accuracy'], target_metrics['accuracy'], rel_tol=1e-6)
    # TODO: validate overall precision, recall, f1
