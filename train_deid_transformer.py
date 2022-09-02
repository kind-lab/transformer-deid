import argparse

import torch
from tqdm import tqdm

from datetime import datetime
import logging
from pathlib import Path
import os
import json
import random
import numpy as np

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification, BertForTokenClassification, RobertaForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric

# local packages
from transformer_deid.data import DeidDataset, DeidTask
from transformer_deid.evaluation import compute_metrics
from transformer_deid.tokenization import assign_tags, encode_tags, split_sequences
from transformer_deid.utils import convert_dict_to_native_types
from transformer_deid.model_evaluation_functions import load_data

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def which_transformer_arch(baseArchitecture):
    if baseArchitecture == 'bert':
        load_model = BertForTokenClassification.from_pretrained
        tokenizerArch = 'bert-base-cased'
        baseArchitecture = 'BERT'

    elif baseArchitecture == 'roberta':
        load_model = RobertaForTokenClassification.from_pretrained
        tokenizerArch = 'roberta-base'
        baseArchitecture = 'RoBERTa'

    elif baseArchitecture == 'distilbert':
        load_model = DistilBertForTokenClassification.from_pretrained
        tokenizerArch = 'distilbert-base-cased'
        baseArchitecture = 'DistilBERT'

    return load_model, tokenizerArch, baseArchitecture


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a transformer-based deid model.'
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
        '-a',
        '--architecture',
        type=str,
        help=
        'name of base architecture, either bert (default), roberta, or distilbert',
        default='bert'
    )

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help='number of epochs; defaults to 1',
        default=1
    )

    args = parser.parse_args()

    return args


def main():
    seed_everything(42)

    args = parse_args()

    task_name = args.task_name
    dataDir = f'./../{task_name}'
    testDir = dataDir + '/test'
    baseArchitecture = args.architecture

    load_model, tokenizerArch, baseArchitecture = which_transformer_arch(
        baseArchitecture
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    deid_task, train_dataset, val_dataset, __ = load_data(
        task_name, dataDir, testDir, tokenizerArch
    )

    model = load_model(tokenizerArch, num_labels=len(deid_task.labels)).to(device)

    epochs = args.epochs
    train_batch_size = 8
    out_dir = f'./{baseArchitecture}results{epochs}'

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='no'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", training_args.num_train_epochs)

    trainer.train()

    save_location = f'{out_dir}/{task_name}_{baseArchitecture}_Model_{epochs}'

    trainer.save_model(save_location)

    trainer.evaluate()


if __name__ == '__main__':
    main()
