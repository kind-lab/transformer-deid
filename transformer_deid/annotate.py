import torch
import argparse
import os
import numpy as np
from pathlib import Path
from data import DeidDataset
from tokenization import convert_subtokens_to_label_list, merge_sequences
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
# from convert_to_label_list import convert_encodings_to_label_list, merge_sequences
from load_data import load_data, create_deid_dataset
from train import which_transformer_arch


def annotate(modelDir: str, test_dataset: DeidDataset):
    """Annotates dataset with PHI labels.

       Args:
            modelDir: directory containing config.json, pytorch_model.bin, and training_args.bin
                e.g., 'i2b2_2014_{base architecture}_Model_{epochs}'
            deid_task: DeidTask, see data.py
            test_dataset: DeidDataset, see data.py

       Returns:
            predicted_label:
            labels:
    """
    # get parameters
    epochs = int(modelDir.split('_')[-1])
    out_dir = '/'.join(modelDir.split('/')[0:-1]) # this is a necessary parameter for some reason
    train_batch_size = 8

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForTokenClassification.from_pretrained(modelDir).to(device)

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
        save_strategy='no')

    trainer = Trainer(model=model,
                      args=training_args)
    
    id2label = model.config.id2label
    predictions, labels, _ = trainer.predict(test_dataset)
    predicted_label = np.argmax(predictions, axis=2)
    predicted_label = [[id2label[token] for token in doc] for doc in predicted_label]

    labels = []
    for i, doc in enumerate(predicted_label):
        labels += [convert_subtokens_to_label_list(doc, test_dataset.encodings[i])]

    new_labels = merge_sequences(labels, test_dataset.ids)


    return new_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Train a transformer-based PHI deidentification model.')

    parser.add_argument(
        '-i',
        '--test_path',
        type=str,
        help=
        'string to diretory containing txt directory for the test set.'
    )

    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        help='model directory')

    parser.add_argument(
        '-o',
        '--output_path',
        help=
        'output path in which to optionally save the annotations', 
        default=None)

    args = parser.parse_args()

    return args

def main(args):
    train_path = args.test_path
    out_path = args.output_path
    modelDir = args.model_path

    baseArchitecture = os.path.basename(modelDir).split('_')[-3].lower()
    __, tokenizerArch, __ = which_transformer_arch(baseArchitecture)
    tokenizer = AutoTokenizer.from_pretrained(tokenizerArch)

    data_dict = load_data(Path(train_path))
    test_dataset = create_deid_dataset(data_dict, tokenizer)

    annotations = annotate(modelDir, test_dataset)
    if out_path is not None:
        # TODO: save annotations in out_path
        raise NotImplementedError('sorry! can\'t save yet!')
    
    else:
        print(annotations[0])
        return annotations

if __name__ == '__main__':
    args = parse_args()
    main(args)