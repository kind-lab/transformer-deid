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


def test_overall_metrics():
    """
    Use specific pre-calculated test cases to validate overall metrics, considering the exclusion of 'O'.
    Note: Accuracy not tested since it directly uses the metrics library.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = TP / (TP + 0.5 * (FP + FN))
    """
    # Multi-class
    predictions1 = np.array(
        [
            [0, 0, 0, 0, 0],
            [3, 3, 2, 1, 1],
            [3, 0, 2, 0, 1]
            ]
    )

    references1 = np.array(
        [
            [0, 0, 1, 0, 0],
            [2, 3, 2, 2, 1],
            [3, 3, 2, 0, 1]
            ]
    ) 
    
    comp_metrics1 = evaluation.compute_metrics(predictions=predictions1, labels=references1, label_list=text_labels, metric=metric, binary_evaluation=False)
    
    # label |  TP  |  FP  |  FN
    #   0   |  -   |  -   |  -
    #   1   |  2   |  1   |  1
    #   2   |  2   |  0   |  2
    #   3   |  2   |  1   |  1
    # total |  6   |  2   |  4 

    TP1 = 6
    FP1 = 2
    FN1 = 4
    precision1 = TP1 / (TP1 + FP1)
    recall1 = TP1 / (TP1 + FN1)
    f1_1 = TP1 / (TP1 + 0.5 * (FP1 + FN1))

    assert math.isclose(comp_metrics1['overall_precision'], precision1, rel_tol=1e-6)
    assert math.isclose(comp_metrics1['overall_recall'], recall1, rel_tol=1e-6)
    assert math.isclose(comp_metrics1['overall_f1'], f1_1, rel_tol=1e-6)


    # Multi-class
    predictions2 = np.array(
        [
            [1, 3, 2, 3, 1],
            [1, 3, 2, 1, 1],
            [3, 0, 2, 0, 2]
            ]
    )

    references2 = np.array(
        [
            [1, 3, 1, 3, 3],
            [2, 3, 2, 2, 1],
            [1, 3, 2, 0, 1]
            ]
    )

    comp_metrics2 = evaluation.compute_metrics(predictions=predictions2, labels=references2, label_list=text_labels, metric=metric, binary_evaluation=False)
    
    # label |  TP  |  FP  |  FN
    #   0   |  -   |  -   |  -
    #   1   |  2   |  3   |  3
    #   2   |  2   |  2   |  2
    #   3   |  3   |  1   |  2
    # total |  7   |  6   |  7

    TP2 = 7
    FP2 = 6
    FN2 = 7
    precision2 = TP2 / (TP2 + FP2)
    recall2 = TP2 / (TP2 + FN2)
    f1_2 = TP2 / (TP2 + 0.5 * (FP2 + FN2))

    assert math.isclose(comp_metrics2['overall_precision'], precision2, rel_tol=1e-6)
    assert math.isclose(comp_metrics2['overall_recall'], recall2, rel_tol=1e-6)
    assert math.isclose(comp_metrics2['overall_f1'], f1_2, rel_tol=1e-6)


    # Binary
    predictions3 = np.array(
        [
            [1, 3, 5, 3, 6],
            [5, 3, 2, 1, 1],
            [5, 0, 2, 0, 7]
            ]
    )

    references3 = np.array(
        [
            [0, 2, 2, 3, 3],
            [0, 3, 0, 2, 0],
            [5, 3, 3, 0, 0]
            ]
    )    
    
    comp_metrics3 = evaluation.compute_metrics(predictions=predictions3, labels=references3, label_list=text_labels, metric=metric, binary_evaluation=True)

    # label |  TP  |  FP  |  FN
    #   0   |  -   |  -   |  -
    # non-0 |  8   |  5   |  1

    TP3 = 8
    FP3 = 5
    FN3 = 1
    precision3 = TP3 / (TP3 + FP3)
    recall3 = TP3 / (TP3 + FN3)
    f1_3 = TP3 / (TP3 + 0.5 * (FP3 + FN3))

    assert math.isclose(comp_metrics3['overall_precision'], precision3, rel_tol=1e-6)
    assert math.isclose(comp_metrics3['overall_recall'], recall3, rel_tol=1e-6)
    assert math.isclose(comp_metrics3['overall_f1'], f1_3, rel_tol=1e-6)
