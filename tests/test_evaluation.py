import numpy as np
import math
from pathlib import Path
from sklearn import metrics
from datasets import load_metric
from transformer_deid import evaluation

metric_dir = f"{Path(__file__).parent}/../transformer_deid/token_evaluation.py"
metric = load_metric(metric_dir)

text_labels = ['O', 'AGE', 'CONTACT', 'DATE', 'ID', 'LOCATION', 'NAME', 'PROFESSION']


def test_individual_class_metrics():
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


def test_individual_binary_metrics():
    """
    Generate 10 test pairs randomly, compare calculated binary metrics to sklearn.metrics
    """
    predictions = np.random.randint(0, high=len(text_labels) - 1, size=(10, 250))
    references = np.random.randint(1, high=len(text_labels) - 1, size=(10, 250))

    binary_predictions = predictions.flatten() == 0
    binary_references = references.flatten() == 0

    comp_metrics = evaluation.compute_metrics(predictions=predictions, labels=references, label_list=text_labels, metric=metric, binary_evaluation=True)
    target_metrics = metrics.classification_report(y_pred=binary_predictions, y_true=binary_references, output_dict=True)

    assert math.isclose(comp_metrics['PHI']['f1'], target_metrics['False']['f1-score'], rel_tol=1e-6)
    assert math.isclose(comp_metrics['PHI']['number'], target_metrics['False']['support'], rel_tol=1e-6)
    assert math.isclose(comp_metrics['PHI']['precision'], target_metrics['False']['precision'], rel_tol=1e-6)
    assert math.isclose(comp_metrics['PHI']['recall'], target_metrics['False']['recall'], rel_tol=1e-6)
    assert math.isclose(comp_metrics['overall_accuracy'], target_metrics['accuracy'], rel_tol=1e-6)


def test_overall_metrics_multiclass1():
    """
    Use specific pre-calculated test cases to validate overall metrics, considering the exclusion of 'O'.
    Note: Accuracy not tested since it directly uses the metrics library.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = TP / (TP + 0.5 * (FP + FN))
    """
    # Multi-class
    predictions = np.array(
        [
            [0, 0, 0, 0, 0],
            [3, 3, 2, 1, 1],
            [3, 0, 2, 0, 1]
            ]
    )

    references = np.array(
        [
            [0, 0, 1, 0, 0],
            [2, 3, 2, 2, 1],
            [3, 3, 2, 0, 1]
            ]
    ) 
    
    comp_metrics = evaluation.compute_metrics(predictions=predictions, labels=references, label_list=text_labels, metric=metric, binary_evaluation=False)
    
    # label |  TP  |  FP  |  FN
    #   0   |  -   |  -   |  -
    #   1   |  2   |  1   |  1
    #   2   |  2   |  0   |  2
    #   3   |  2   |  1   |  1
    # total |  6   |  2   |  4 

    TP = 6
    FP = 2
    FN = 4
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = TP / (TP + 0.5 * (FP + FN))

    assert math.isclose(comp_metrics['overall_precision'], precision, rel_tol=1e-6)
    assert math.isclose(comp_metrics['overall_recall'], recall, rel_tol=1e-6)
    assert math.isclose(comp_metrics['overall_f1'], f1, rel_tol=1e-6)


def test_overall_metrics_multiclass2():
    """
    Use specific pre-calculated test cases to validate overall metrics, considering the exclusion of 'O'.
    Note: Accuracy not tested since it directly uses the metrics library.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = TP / (TP + 0.5 * (FP + FN))
    """
    # Multi-class
    predictions = np.array(
        [
            [1, 3, 2, 3, 1],
            [1, 3, 2, 1, 1],
            [3, 0, 2, 0, 2]
            ]
    )

    references = np.array(
        [
            [1, 3, 1, 3, 3],
            [2, 3, 2, 2, 1],
            [1, 3, 2, 0, 1]
            ]
    )

    comp_metrics = evaluation.compute_metrics(predictions=predictions, labels=references, label_list=text_labels, metric=metric, binary_evaluation=False)
    
    # label |  TP  |  FP  |  FN
    #   0   |  -   |  -   |  -
    #   1   |  2   |  3   |  3
    #   2   |  2   |  2   |  2
    #   3   |  3   |  1   |  2
    # total |  7   |  6   |  7

    TP = 7
    FP = 6
    FN = 7
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = TP / (TP + 0.5 * (FP + FN))

    assert math.isclose(comp_metrics['overall_precision'], precision, rel_tol=1e-6)
    assert math.isclose(comp_metrics['overall_recall'], recall, rel_tol=1e-6)
    assert math.isclose(comp_metrics['overall_f1'], f1, rel_tol=1e-6)


def test_overall_metrics_binary():
    """
    Use specific pre-calculated test cases to validate overall metrics, considering the exclusion of 'O'.
    Note: Accuracy not tested since it directly uses the metrics library.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = TP / (TP + 0.5 * (FP + FN))
    """
    # Binary
    predictions = np.array(
        [
            [1, 3, 5, 3, 6],
            [5, 3, 2, 1, 1],
            [5, 0, 2, 0, 7]
            ]
    )

    references = np.array(
        [
            [0, 2, 2, 3, 3],
            [0, 3, 0, 2, 0],
            [5, 3, 3, 0, 0]
            ]
    )    
    
    comp_metrics = evaluation.compute_metrics(predictions=predictions, labels=references, label_list=text_labels, metric=metric, binary_evaluation=True)

    # label |  TP  |  FP  |  FN
    #   0   |  -   |  -   |  -
    # non-0 |  8   |  5   |  1

    TP = 8
    FP = 5
    FN = 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = TP / (TP + 0.5 * (FP + FN))

    assert math.isclose(comp_metrics['overall_precision'], precision, rel_tol=1e-6)
    assert math.isclose(comp_metrics['overall_recall'], recall, rel_tol=1e-6)
    assert math.isclose(comp_metrics['overall_f1'], f1, rel_tol=1e-6)
