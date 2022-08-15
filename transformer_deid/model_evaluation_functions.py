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
    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_deid_dataset(texts, labels, tokenizer,
                        label2id: dict) -> DeidDataset:
    """Creates a dataset from set of texts and labels.

       Args:
            texts: dict of text data, from, e.g., DeidTask.train['text']
            labels: dict of annotations, from, e.g., DeidTask.train['ann']
            tokenizer: HuggingFace tokenizer, e.g., loaded from AutoTokenizer.from_pretrained()
            label2id: dict property of a DeidTask (see data.py)

       Returns:
            DeidDataset; see class definition in data.py
    """

    # specify dataset arguments
    split_long_sequences = True

    # split text/labels into multiple examples
    # (1) tokenize text
    # (2) identify split points
    # (3) output text as it was originally
    if split_long_sequences:
        texts, labels = split_sequences(tokenizer, texts, labels)

    encodings = tokenizer(texts,
                          is_split_into_words=False,
                          return_offsets_mapping=True,
                          padding=True,
                          truncation=True)

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
    """Creates a DeidTask; loads the training, validation, and test data.

       Args:
            task_name: origin of training data, e.g., 'i2b2_2014'
            dataDir: directory with training data containing two folders ('txt' and 'ann')
                divided 80-20 between training and validation sets
            testDir: directory with testing data containing two folders ('txt' and 'ann')
            tokenizerArch: name of HuggingFace pretrained tokenizer
                e.g., 'bert-base-cased'

       Returns:
            deid_task: DeidTask, see data.py
            train_dataset: DeidDataset of training data, generally 80% of train set
            val_dataset: DeidDataset of validation data, generally 20% of train set
            test_dataset: DeidDataset of training data
    """

    # specify dataset arguments
    label_transform = 'base'

    # definition in data.py
    deid_task = DeidTask(
        task_name,
        # data_dir=f'/home/alistairewj/git/deid-gs/{task_name}',
        data_dir=dataDir,
        test_dir=testDir,
        label_transform=label_transform)

    train_texts, train_labels = deid_task.train['text'], deid_task.train['ann']
    split_idx = int(0.8 * len(train_texts))
    val_texts, val_labels = train_texts[split_idx:], train_labels[split_idx:]
    train_texts, train_labels = train_texts[:split_idx], train_labels[:split_idx]
    test_texts, test_labels = deid_task.test['text'], deid_task.test['ann']

    tokenizer = AutoTokenizer.from_pretrained(tokenizerArch)
    label2id = deid_task.label2id

    train_dataset = create_deid_dataset(train_texts, train_labels, tokenizer,
                                        label2id)
    val_dataset = create_deid_dataset(val_texts, val_labels, tokenizer,
                                      label2id)
    test_dataset = create_deid_dataset(test_texts, test_labels, tokenizer,
                                       label2id)

    return deid_task, train_dataset, val_dataset, test_dataset


def load_new_test_set(deid_task, newTestPath: str, tokenizerArch: str):
    """Sets a new dataset as the test set in the DeidTask.

       Args:
            deid_task: DeidTask to be changed, see data.py
            newTestPath: directory to new test data containing txt and ann folders
            tokenizerArch: name of HuggingFace pretrained tokenizer
                e.g., 'bert-base-cased'

       Returns:
            deid_task: modified DeidTask
            test_dataset: DeidDataset corresponding to new directory
    """

    deid_task.set_test_set(newTestPath)
    test_texts, test_labels = deid_task.test['text'], deid_task.test['ann']
    tokenizer = AutoTokenizer.from_pretrained(tokenizerArch)

    test_dataset = create_deid_dataset(test_texts, test_labels, tokenizer,
                                       deid_task.label2id)

    return deid_task, test_dataset


def eval_model(modelDir: str, deid_task: DeidTask, train_dataset: DeidDataset,
               val_dataset: DeidDataset, test_dataset: DeidDataset):
    """Generates all metrics for single a model making inferences on a single dataset.

       Args:
            modelDir: directory containing config.json, pytorch_model.bin, and training_args.bin
                e.g., 'i2b2_2014_{base architecture}_Model_{epochs}'
            deid_task: DeidTask, see data.py
            train, val, and test_dataset: DeidDatasets, see data.py

       Returns:
            results_multiclass: dict of operating point statistics (precision, recall, f1) for each datatype
                datatypes are: age, contact, date, ID, location, name, profession
            results_binary: dict of operating point statistics (precision, recall, f1) for binary label
                i.e., PHI or non-PHI labels
    """

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
        save_strategy='epoch')

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset)

    predictions, labels, _ = trainer.predict(test_dataset)
    predicted_label = np.argmax(predictions, axis=2)

    # load metric to be used -- if none is passed, default to seqeval
    metric_dir = "transformer_deid/token_evaluation.py"
    metric = load_metric(metric_dir)

    results_multiclass = compute_metrics(predicted_label,
                                         labels,
                                         deid_task.labels,
                                         metric=metric)

    results_binary = compute_metrics(predicted_label,
                                     labels,
                                     deid_task.labels,
                                     metric=metric,
                                     binary_evaluation=True)

    return results_multiclass, results_binary


# function to return all metrics in two lists


def eval_model_list(modelDirList: list,
                    dataDir: str,
                    testDirList: list,
                    output_metric=None) -> list:
    """Generate all metrics or a specific metric for a list of models over all test sets in a list of test sets

       Args:
            modelDirList: list of model directories
                each modelDir should have the form 'i2b2_2014_{base architecture}_Model_{epochs}'
            dataDir: directory containing training/validation data with txt and ann folders
            testDirList: list of test data directories
                each test dir must have txt and ann folders
            output_metric: name of metric to be returned from binary evaluation
                if None, returns all metrics (multiclass and binary)

       Returns:
            results: list of lists, each entry corresponding to the output_metric argument
                for a model's inference on a test dataset
    """

    results = []

    for j, modelDir in enumerate(modelDirList):
        baseArchitecture = modelDir.split('_')[-3].lower()
        task_name = dataDir.split('/')[-1]

        if baseArchitecture == 'bert':
            tokenizerArch = 'bert-base-cased'
        elif baseArchitecture == 'roberta':
            tokenizerArch = 'roberta-base'
        elif baseArchitecture == 'distilbert':
            tokenizerArch = 'distilbert-base-cased'

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
