import argparse
import math
import pprint

from datetime import datetime
import logging
from pathlib import Path
import os
import json
from tqdm import tqdm

import numpy as np

from transformers import BertTokenizerFast
from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric

# local packages
from transformer_deid.data import DeidDataset, DeidTask
from transformer_deid.evaluation import compute_metrics
from transformer_deid.tokenization import assign_tags, encode_tags, split_sequences
from transformer_deid.utils import convert_dict_to_native_types

from train_deid_transformer import which_transformer_arch
from transformer_deid.model_evaluation_functions import load_data

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

multi_class_fields = [
    'AGEprecision', 'AGErecall', 'AGEf1', 'AGEnumber', 'CONTACTprecision',
    'CONTACTrecall', 'CONTACTf1', 'CONTACTnumber', 'DATEprecision',
    'DATErecall', 'DATEf1', 'DATEnumber', 'IDprecision', 'IDrecall', 'IDf1',
    'IDnumber', 'LOCATIONprecision', 'LOCATIONrecall', 'LOCATIONf1',
    'LOCATIONnumber', 'NAMEprecision', 'NAMErecall', 'NAMEf1', 'NAMEnumber',
    'PROFESSIONprecision', 'PROFESSIONrecall', 'PROFESSIONf1',
    'PROFESSIONnumber', 'overall_precision', 'overall_recall', 'overall_f1',
    'overall_accuracy'
]
binary_fields = [
    'PHIprecision', 'PHIrecall', 'PHIf1', 'PHInumber', 'overall_precision',
    'overall_recall', 'overall_f1', 'overall_accuracy'
]


def flatten_dict(d):
    """
    Return flattened version of the evaluation result dict
    """
    out = {}
    for key in d:
        if type(d[key]) is dict:
            child = flatten_dict(d[key])
            for child_key in child:
                val = child[child_key]
                if isinstance(val, np.int64):
                    val = int(val)
                out[key + child_key] = val
        else:
            out[key] = d[key]
    return out


def add_row(
    path, epochs, results_multiclass, results_binary, multi_class_fields,
    binary_fields, test_loss
):
    """
    Add row to worksheet
    fields: [epochs] + multi_class_fields + binary_fields
    """
    root = Path(path).parent

    row = [epochs] + [
        flatten_dict(results_multiclass).get(field)
        for field in multi_class_fields
    ] + [flatten_dict(results_binary).get(field)
         for field in binary_fields] + [test_loss]

    text_metrics = ','.join(map(str, row)) + '\n'

    with open(str(root) + '/training_eval.csv', 'at') as f:
        f.write(text_metrics)
    # worksheet.append_row(row, table_range='A1')


def eval_checkpoints(
    path, deid_task, train_dataset, val_dataset, test_dataset, training_args
):
    step = int(path.split('-')[-1])
    steps_per_epoch = math.ceil(
        len(train_dataset) / training_args.per_device_train_batch_size
    )
    epoch = step / steps_per_epoch

    model = AutoModelForTokenClassification.from_pretrained(
        path, num_labels=len(deid_task.labels)
    )

    model.eval()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    predictions, labels, metrics = trainer.predict(test_dataset)
    predicted_label = np.argmax(predictions, axis=2)

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
        binary_evaluation=True
    )

    add_row(
        path, epoch, results_multiclass, results_binary, multi_class_fields,
        binary_fields, metrics['test_loss']
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate transformer-based model at each checkpoint.'
    )

    parser.add_argument(
        '-n',
        '--task_name',
        type=str,
        help=
        'name of folder containing train and test data; defaults to i2b2_2014',
        default='i2b2_2014'
    )

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='folder containing checkpoint files',
        default='bert'
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    root = f'{args.model}'
    arch = args.model.split('results')[0].lower()
    epochs = int(args.model.split('results')[1])
    task_name = args.task_name

    _, tokenizerArch, _ = which_transformer_arch(arch)

    dataDir = f'{task_name}'
    testDir = f'{task_name}/test'

    deid_task, train_dataset, val_dataset, test_dataset = load_data(
        task_name, dataDir, testDir, tokenizerArch
    )

    train_batch_size = 8

    training_args = TrainingArguments(
        output_dir=root,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='steps',
        eval_steps=1155
    )
    
    if not os.path.exists(str(root) + '/training_eval.csv'):
        with open(str(root) + '/training_eval.csv', 'wt') as f:
            header = 'epoch,' + ','.join(
                map(str, multi_class_fields + binary_fields + ['test_loss'])
            ) + '\n'
            f.write(header)

    checkpoints = [
        item for item in os.listdir(root)
        if 'checkpoint' in item and os.path.isdir(os.path.join(root, item))
    ]

    for item in tqdm(sorted(checkpoints, key=lambda x: int(x.split('-')[1]))):
        path = os.path.join(root, item)
        eval_checkpoints(
            path, deid_task, train_dataset, val_dataset, test_dataset,
            training_args
        )


if __name__ == '__main__':
    main()
