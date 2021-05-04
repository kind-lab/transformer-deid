from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification
from transformers import Trainer, TrainingArguments

# local packages
from transformer_deid.data import DeidDataset, DeidTask
from transformer_deid.tokenization import assign_tags, encode_tags


def main():
    task_name = 'i2b2_2014'
    deid_task = DeidTask(
        task_name, data_dir=f'/home/alistairewj/git/deid-gs/{task_name}'
    )

    train_texts, train_labels = deid_task.train['text'], deid_task.train['ann']
    test_texts, test_labels = deid_task.test['text'], deid_task.test['ann']

    # create a validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=.2
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    # TODO: ensure we split longer documents
    # few different ways to do this
    #   - split long documents into enough sub-docs to fit everything
    #   - randomly sample from a given document
    #   - same as before and up weight longer documents in selecting examples for batch

    train_encodings = tokenizer(
        train_texts,
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )
    val_encodings = tokenizer(
        val_texts,
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )

    # use the offset mappings in train_encodings to assign labels to tokens
    train_tags = assign_tags(train_encodings, train_labels)
    val_tags = assign_tags(val_encodings, val_labels)

    # encodings are dicts with three elements:
    #   'input_ids', 'attention_mask', 'offset_mapping'
    # these are used as kwargs to model training later
    train_labels = encode_tags(train_tags, train_encodings, deid_task.label2id)
    val_labels = encode_tags(val_tags, val_encodings, deid_task.label2id)

    # prepare a dataset compatible with Trainer module
    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")
    train_dataset = DeidDataset(train_encodings, train_labels)
    val_dataset = DeidDataset(val_encodings, val_labels)

    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-cased', num_labels=len(deid_task.labels)
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    trainer.save_model(f'results/{task_name}_DistilBert_Model')

    trainer.evaluate()


if __name__ == '__main__':
    main()
