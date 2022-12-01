import torch
import random
import os
import numpy as np
import logging
import argparse
from pathlib import Path
from transformers import DistilBertForTokenClassification, BertForTokenClassification, RobertaForTokenClassification, AutoModelForTokenClassification, AutoConfig
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from load_data import create_deid_dataset, get_labels, load_data

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
    else:
        raise NotImplementedError(f'{baseArchitecture} not a recognized model')

    return load_model, tokenizerArch, baseArchitecture

def train(train_data_dict, architecture, epochs):
    out_dir = '../test_save/'
    seed_everything(42)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    load_model, tokenizerArch, baseArchitecture = which_transformer_arch(architecture)
    tokenizer = AutoTokenizer.from_pretrained(tokenizerArch)

    unique_labels = get_labels(train_data_dict['ann'])
    label2id = {tag: id for id, tag in enumerate(unique_labels)}
    id2label = {id: tag for tag, id in label2id.items()}

    train_dataset = create_deid_dataset(train_data_dict, tokenizer, label2id)
    config = AutoConfig.from_pretrained(tokenizerArch, num_labels=len(unique_labels))#, label2id=label2id, id2label=id2label)

    model = load_model(tokenizerArch, num_labels=len(unique_labels), label2id=label2id, id2label=id2label).to(device)

    train_batch_size = 8

    training_args = TrainingArguments(
        output_dir=out_dir, # this is a mandatory argument
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='no', # probably make none? or make a parameter?
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", training_args.num_train_epochs)

    trainer.train()

    return trainer

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Train a transformer-based PHI deidentification model.')

    parser.add_argument(
        '-i',
        '--train_path',
        type=str,
        help=
        'string to diretory containing txt and ann directories for the training set.'
    )

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help='number of epochs to train over',
        default=5)

    parser.add_argument(
        '-m',
        '--model_architecture',
        type=str,
        choices=['bert', 'distilbert', 'roberta'],
        help='name of model architecture, either bert, roberta, or distilbert',
        default='bert')

    parser.add_argument(
        '-o',
        '--output_path',
        help=
        'output path in which to save the model')

    args = parser.parse_args()

    return args

def main(args):
    # arguments
    train_path = args.train_path
    out_path = args.output_path
    model = args.model_architecture
    epochs = args.epochs

    data_dict = load_data(Path(train_path))

    trainer = train(data_dict, model, epochs)

    save_location = f'{out_path}/{model}_model_{epochs}'

    trainer.save_model (save_location)


if __name__ == '__main__':
    args = parse_args()
    main(args)