import argparse
import logging
from pathlib import Path
from xmlrpc.client import Boolean
from datasets import load_metric
import os
import csv
import json
import pprint
from importlib.resources import open_text
from transformers import AutoTokenizer

from transformer_deid import model_evaluation_functions as eval
from transformer_deid.label import Label
from transformer_deid.evaluation import compute_metrics
from transformer_deid.utils import convert_dict_to_native_types

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
_LOGGER = logging.getLogger(__name__)


def get_label_map(transform):
    with open_text('transformer_deid', 'label.json') as fp:
        label_map = json.load(fp)

    # label_membership has different label transforms as keys
    if transform not in label_map:
        raise KeyError('Unable to find label transform %s in label.json' %
                       transform)
    label_map = label_map[transform]

    # label_map has items "harmonized_label": ["label 1", "label 2", ...]
    # invert this for the final label mapping
    return {
        label: harmonized_label
        for harmonized_label, original_labels in label_map.items()
        for label in original_labels
    }


def load_label(filename, label_map):
    """
    Loads annotations from a CSV file.
    CSV file should have entity_type/start/stop columns.
    """
    with open(filename, 'r') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        header = next(csvreader)
        # identify which columns we want
        idx = [
            header.index('entity_type'),
            header.index('start'),
            header.index('stop'),
            header.index('entity')
        ]

        # iterate through the CSV and load in the labels
        if label_map is not None:
            labels = [
                Label(entity_type=label_map[row[idx[0]].upper()],
                      start=int(row[idx[1]]),
                      length=int(row[idx[2]]) - int(row[idx[1]]),
                      entity=row[idx[3]]) for row in csvreader
            ]
        else:
            labels = [
                Label(entity_type=row[idx[0].upper()],
                      start=int(row[idx[1]]),
                      length=int(row[idx[2]]) - int(row[idx[1]]),
                      entity=row[idx[3]]) for row in csvreader
            ]

    return labels


def load_data(path, file_ext, label_map) -> dict:
    """Creates a dict the dataset."""
    examples = {'guid': [], 'text': [], 'ann': []}

    # for deid datasets, "path" is a folder containing txt/ann subfolders
    # "txt" subfolder has text files with the text of the examples
    # "ann" subfolder has annotation files with the labels
    txt_path = path / 'txt'
    ann_path = path / file_ext
    for f in os.listdir(txt_path):
        if not f.endswith('.txt'):
            continue

        # guid is the file name
        guid = f[:-4]
        with open(txt_path / f, 'r') as fp:
            text = ''.join(fp.readlines())

        # load the annotations from disk
        # these datasets have consistent folder structures:
        #   root_path/txt/RECORD_NAME.txt - has text
        #   root_path/ann/RECORD_NAME.gs - has annotations
        if file_ext.endswith('ann'):
            labels = load_label(ann_path / f'{f[:-4]}.gs', label_map)

        elif file_ext.endswith('output'):
            labels = load_label(ann_path / f'{f[:-4]}.ann', label_map)

        examples['guid'].append(guid)
        examples['text'].append(text)
        examples['ann'].append(labels)

    return examples


def get_labels(ann_dict):
    """Gets the list of labels for this data set."""
    unique_labels = set(label.entity_type for labels in ann_dict
                        for label in labels)

    # add in test set labels
    # unique_labels_test = set(label.entity_type for labels in self.test['ann'] for label in labels)
    # unique_labels = list(unique_labels.union(unique_labels_test))
    unique_labels = list(unique_labels)

    unique_labels.sort()
    # add in the object tag - ensure it is the first element for label2id and id2label dict
    if 'O' in unique_labels:
        unique_labels.pop('O')
    unique_labels = ['O'] + unique_labels

    return unique_labels


def compare_annotations(path, predictions=None, actual=None):
    """Returns metrics comparing two sets of annotations. 

       Args:
            path: string to directory containing folders of both gold standard and predicted annotations
            predictions: optional string, name of directory with predicted annotations
                defaults to 'output'
            actual: optional string, name of directory with predicted annotations
                defaults to 'ann', consistent with i2b2 data
        
        Returns: 
            results_multiclass: dict of operating point statistics (precision, recall, f1) for each datatype
                datatypes are: age, contact, date, ID, location, name, profession
            results_binary: dict of operating point statistics (precision, recall, f1) for binary label
                i.e., PHI or non-PHI labels

    """

    # generate label map from label.json
    label_map = get_label_map('base')

    path = Path(path)

    # create dictionaries of both sets of annotations
    if predictions is not None:
        output_dict = load_data(path, predictions, label_map)
    else:
        output_dict = load_data(path, 'output', label_map)

    if actual is not None:
        actual_dict = load_data(path, actual, label_map)
    else:
        actual_dict = load_data(path, 'ann', label_map)

    # set tokenizer
    tokenizerArch = 'bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(tokenizerArch)

    # generate DeidDatasets for both sets of annotations
    pred_labels = get_labels(output_dict['ann'])
    label2id = {tag: id for id, tag in enumerate(pred_labels)}
    output_dataset = eval.create_deid_dataset(output_dict['text'],
                                              output_dict['ann'], tokenizer,
                                              label2id)

    labels = get_labels(actual_dict['ann'])
    label2id = {tag: id for id, tag in enumerate(labels)}
    actual_dataset = eval.create_deid_dataset(actual_dict['text'],
                                              actual_dict['ann'], tokenizer,
                                              label2id)

    predicted_label = output_dataset.labels
    real_labels = actual_dataset.labels

    # use in-house metric
    metric_dir = "transformer_deid/token_evaluation.py"
    metric = load_metric(metric_dir)

    predicted_label = eval.decode_labels(predicted_label,
                                         pred_labels,
                                         true_labels=real_labels)
    real_labels = eval.decode_labels(real_labels, labels)

    results_multiclass = compute_metrics(predicted_label,
                                         real_labels,
                                         metric=metric)

    results_binary = compute_metrics(predicted_label,
                                     real_labels,
                                     metric=metric,
                                     binary_evaluation=True)

    return results_multiclass, results_binary


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Compare gold-standard annotations to those generated by pydeid')

    parser.add_argument(
        '-p',
        '--path',
        type=str,
        help=
        'string to diretory containing folders of both gold standard and predicted annotations'
    )

    parser.add_argument(
        '-e',
        '--predictions',
        type=str,
        help='name of directory with predicted annotations; defaults to output',
        default=None)

    parser.add_argument(
        '-a',
        '--actual',
        type=str,
        help='name of directory with predicted annotations; defaults to ann',
        default=None)

    parser.add_argument(
        '-o',
        '--output',
        type=Boolean,
        help=
        'if True, create .json of outputs; if False (default) output to stdout',
        default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    path = args.path
    predictions = args.predictions
    actual = args.actual

    results_multiclass, results_binary = compare_annotations(
        path, predictions=predictions, actual=actual)

    output = args.output

    if output:
        results = {
            'results_multiclass':
            convert_dict_to_native_types(results_multiclass),
            'results_binary': convert_dict_to_native_types(results_binary)
        }

        with open(path + f'/{predictions}_results.json', 'w') as outfile:
            json.dump(results, outfile, indent=4)

    else:
        print('\nMulti-class results:')
        pprint.pprint(results_multiclass)
        print('\nBinary results')
        pprint.pprint(results_binary)


if __name__ == '__main__':
    main()
