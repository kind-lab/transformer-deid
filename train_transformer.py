from datetime import datetime
import logging
from pathlib import Path
import os
import json

import numpy as np

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric

# local packages
from transformer_deid.data import DeidDataset, DeidTask
from transformer_deid.evaluation import compute_metrics
from transformer_deid.tokenization import assign_tags, encode_tags, split_sequences


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    # specify dataset arguments
    task_name = 'i2b2_2014'
    split_long_sequences = True
    label_transform = 'base'

    deid_task = DeidTask(
        task_name,
        data_dir=f'/home/alistairewj/git/deid-gs/{task_name}',
        label_transform=label_transform
    )

    train_texts, train_labels = deid_task.train['text'], deid_task.train['ann']
    test_texts, test_labels = deid_task.test['text'], deid_task.test['ann']

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

    # split text/labels into multiple examples
    # (1) tokenize text
    # (2) identify split points
    # (3) output text as it was originally
    if split_long_sequences:
        train_texts, train_labels = split_sequences(train_texts, train_labels, tokenizer)
        test_texts, test_labels = split_sequences(test_texts, test_labels, tokenizer)

    train_encodings = tokenizer(
        train_texts,
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )
    test_encodings = tokenizer(
        test_texts,
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )

    # use the offset mappings in train_encodings to assign labels to tokens
    train_tags = assign_tags(train_encodings, train_labels)
    test_tags = assign_tags(test_encodings, test_labels)

    # encodings are dicts with three elements:
    #   'input_ids', 'attention_mask', 'offset_mapping'
    # these are used as kwargs to model training later
    train_labels = encode_tags(train_tags, train_encodings, deid_task.label2id)
    test_labels = encode_tags(test_tags, test_encodings, deid_task.label2id)

    # prepare a dataset compatible with Trainer module
    train_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")
    train_dataset = DeidDataset(train_encodings, train_labels)
    test_dataset = DeidDataset(test_encodings, test_labels)

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
        eval_dataset=test_dataset
    )


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", training_args.num_train_epochs)
    
    # log top 5 examples
    for i in range(min(len(train_dataset), 5)):
        input_ids, attention_mask, token_type_ids, label_ids = train_dataset.get_example(
            i, deid_task.id2label
        )

        # convert ids into human interpretable values
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        labels = ['-100' if l == -100 else deid_task.id2label[l] for l in label_ids]

        logger.info("*** Example %d ***", i)
        logger.info("tokens: %s", " ".join(tokens))
        logger.info("labels: %s", " ".join(labels))
        logger.info("input_ids: %s", " ".join(map(str, input_ids)))
        logger.info("label_ids: %s", " ".join(map(str, label_ids)))
        logger.info("input_mask: %s", " ".join(map(str, attention_mask)))

    trainer.train()

    trainer.save_model(f'results/{task_name}_DistilBert_Model')

    trainer.evaluate()

    predictions, labels, _ = trainer.predict(test_dataset)
    predicted_label = np.argmax(predictions, axis=2)

    curr_dir = Path(__file__).parent
    metric_dir = str((curr_dir / "transformer_deid/token_evaluation.py").absolute())
    metric = load_metric(metric_dir)
    results = compute_metrics(predictions, labels, deid_task.labels, metric=metric)

    # output results to a log file
    if not os.path.exists(training_args.output_dir):
        os.mkdirs(training_args.output_dir)
    
    # add a few arguments to the results json
    result_time = datetime.now().strftime('%Y-%m-%dT%H%M%S')
    results['params'] = {
        'task_name': task_name,
        'label_transform': label_transform,
        'split_long_sequences': split_long_sequences,
    }
    with open(os.path.join(training_args.output_dir, f'{result_time}_{task_name}_DistilBert.json'), 'w') as fp:
        json.dump(results, fp)
    
    print(results)


if __name__ == '__main__':
    main()
