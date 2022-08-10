import torch
from tqdm import tqdm

from datetime import datetime
import logging
from pathlib import Path
import os
import json
import pprint
import math

import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric

# local packages
from transformer_deid.data import DeidDataset, DeidTask
from transformer_deid.evaluation import compute_metrics
from transformer_deid.tokenization import assign_tags, encode_tags, split_sequences
from transformer_deid.utils import convert_dict_to_native_types

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def createDeidDataset(texts, labels, tokenizer, label2id: dict) -> DeidDataset:
    """Create a dataset from set of texts and labels
    note: label2id is a property of a DeidTask"""

    # specify dataset arguments
    split_long_sequences = True

    # split text/labels into multiple examples
    # (1) tokenize text
    # (2) identify split points
    # (3) output text as it was originally
    if split_long_sequences:
        texts, labels = split_sequences(
            tokenizer, texts, labels
        )

    encodings = tokenizer(
        texts,
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )

    # use the offset mappings in train_encodings to assign labels to tokens
    tags = assign_tags(encodings, labels)

    # encodings are dicts with three elements:
    #   'input_ids', 'attention_mask', 'offset_mapping'
    # these are used as kwargs to model training later
    tags = encode_tags(tags, encodings, label2id)

    # prepare a dataset compatible with Trainer module
    encodings.pop("offset_mapping")
    dataset = DeidDataset(encodings, tags)

    return dataset


def load_data(task_name, dataDir, testDir, tokenizerArch: str):
    """Create a DeidTask; load the training and validation data from dataDir; load the test data from testDir"""

    # specify dataset arguments
    label_transform = 'base'

    # definition in data.py
    deid_task = DeidTask(
        task_name,
        # data_dir=f'/home/alistairewj/git/deid-gs/{task_name}',
        data_dir=dataDir,
        test_dir=testDir,
        label_transform=label_transform
    )

    train_texts, train_labels = deid_task.train['text'], deid_task.train['ann']
    split_idx = int(0.8 * len(train_texts))
    val_texts, val_labels = train_texts[split_idx:], train_labels[split_idx:]
    train_texts, train_labels = train_texts[:split_idx], train_labels[:split_idx]
    test_texts, test_labels = deid_task.test['text'], deid_task.test['ann']

    tokenizer = AutoTokenizer.from_pretrained(tokenizerArch)
    label2id = deid_task.label2id

    train_dataset = createDeidDataset(
        train_texts, train_labels, tokenizer, label2id)
    val_dataset = createDeidDataset(val_texts, val_labels, tokenizer, label2id)
    test_dataset = createDeidDataset(
        test_texts, test_labels, tokenizer, label2id)

    return deid_task, train_dataset, val_dataset, test_dataset


def load_new_test_set(deid_task, newTestPath: str, tokenizerArch: str):
    """Set a new dataset as the test set in the DeidTask"""

    deid_task.set_test_set(newTestPath)
    test_texts, test_labels = deid_task.test['text'], deid_task.test['ann']
    tokenizer = AutoTokenizer.from_pretrained(tokenizerArch)

    test_dataset = createDeidDataset(
        test_texts,
        test_labels,
        tokenizer,
        deid_task.label2id)

    return deid_task, test_dataset


def eval_model(modelDir, deid_task, train_dataset, val_dataset, test_dataset):
    """Generate all metrics for single a model making inferences on a single dataset"""

    epochs = int(modelDir.split('_')[-1])
    out_dir = '/'.join(modelDir.split('/')[0:-1])
    train_batch_size = 8

    model = AutoModelForTokenClassification.from_pretrained(
        modelDir, num_labels=len(deid_task.labels))

    model.eval()

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    predictions, labels, _ = trainer.predict(test_dataset)
    predicted_label = np.argmax(predictions, axis=2)

    # load metric to be used -- if none is passed, default to seqeval
    metric_dir = "transformer_deid/token_evaluation.py"
    metric = load_metric(metric_dir)

    results_multiclass = compute_metrics(
        predicted_label, labels, deid_task.labels, metric=metric
    )

    results_binary = compute_metrics(
        predicted_label,
        labels,
        deid_task.labels,
        metric=metric,
        binary_evaluation=True)

    return results_multiclass, results_binary

# function to return all metrics in two lists


def eval_model_list(modelDirList, dataDir, testDirList, output_metric=None):
    """Generate all metrics or a specific metric for a list of models over all test sets in a list of test sets"""

    results = []

    for j, modelDir in enumerate(modelDirList):
        baseArchitecture = modelDir.split('_')[-3].lower()
        task_name = dataDir.split('/')[-1]

        if baseArchitecture == 'bert':
            tokenizerArch = 'bert-base-cased'
        elif baseArchitecture == 'roberta':
            tokenizerArch = 'roberta-base'

        modelResults = []

        for i, testDir in enumerate(testDirList):
            if (i == 0) and (j == 0):
                deid_task, train_dataset, val_dataset, test_dataset = load_data(
                    task_name, dataDir, testDir, tokenizerArch)
            else:
                deid_task, test_dataset = load_new_test_set(
                    deid_task, testDir, tokenizerArch)

            results_multiclass, results_binary = eval_model(
                modelDir, deid_task, train_dataset, val_dataset, test_dataset)

            if output_metric is None:
                modelResults += [[results_multiclass, results_binary]]
            else:
                modelResults += [results_binary[output_metric]]

        results += [modelResults]

    return results
