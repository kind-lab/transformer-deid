"""Methods for evaluating model performance."""

def compute_metrics(predictions, labels, label_list, metric=load_metric("seqeval"), binary_evaluation=False) -> dict:
    """Returns a dictionary of operating point statistics (Precision/Recall/F1)."""

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    if binary_evaluation:
        # convert all labels to PHI or not PHI
        true_predictions = [
            ['PHI' if w != 'O' else 'O' for w in sequence]
            for sequence in true_predictions
        ]
        true_labels = [
            ['PHI' if w != 'O' else 'O' for w in sequence]
            for sequence in true_labels
        ]
    
    # convert to BIOC
    # true_predictions = convert_to_bio_scheme(true_predictions)
    # true_labels = convert_to_bio_scheme(true_labels)

    return metric.compute(predictions=true_predictions, references=true_labels)
