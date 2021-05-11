from pathlib import Path
from typing import List, Optional, Union, TextIO
from importlib.resources import open_text
import os
from collections import OrderedDict
import csv
import json

import torch

from transformer_deid.label import Label

class DeidTask(object):
    """Utility class for loading and preparing dataset from disk."""
    def __init__(self, task_name, data_dir, label_transform=None):
        """Initialize a data processor with the location of the data."""
        self.task_name = task_name
        self.data_dir = Path(data_dir)

        # transform the labels - used in the label extraction function
        if label_transform is not None:
            self.label_map = self.get_label_map(label_transform)
        else:
            self.label_map = None
        
        # load the train/test data from file
        # these are lists of dicts, each dict has three keys: guid, text, tags
        self.train = self.load_data(self.data_dir / 'train')
        self.test = self.load_data(self.data_dir / 'test')

        # create a list of unique labels
        self.labels = self.get_labels()
        self.label2id = {tag: id for id, tag in enumerate(self.labels)}
        self.id2label = {id: tag for tag, id in self.label2id.items()}

    def get_label_map(self, transform):
        with open_text('transformer_deid', 'label.json') as fp:
            label_map = json.load(fp)
        
        # label_membership has different label transforms as keys
        if transform not in label_map:
            raise KeyError('Unable to find label transform %s in label.json' % transform)
        label_map = label_map[transform]
    
        # label_map has items "harmonized_label": ["label 1", "label 2", ...]
        # invert this for the final label mapping
        return {
            label: harmonized_label
            for harmonized_label, original_labels in label_map.items()
            for label in original_labels
        }

    def get_labels(self):
        """Gets the list of labels for this data set."""
        unique_labels = set(label.entity_type for labels in self.train['ann'] for label in labels)

        # add in test set labels
        unique_labels_test = set(label.entity_type for labels in self.test['ann'] for label in labels)
        unique_labels = list(unique_labels.union(unique_labels_test))
        
        unique_labels.sort()
        # add in the object tag - ensure it is the first element for label2id and id2label dict
        if 'O' in unique_labels:
            unique_labels.pop('O')
        unique_labels = ['O'] + unique_labels

        return unique_labels

    def _read_file(self, input_file, delimiter=',', quotechar='"'):
        """Reads a comma separated value file."""
        fn = os.path.join(self.data_dir, input_file)
        with open(fn, "r") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return self._read_file(input_file, delimiter='\t', quotechar=quotechar)

    def _read_csv(self, input_file, quotechar='"'):
        """Reads a comma separated value file."""
        return self._read_file(input_file, delimiter=',', quotechar=quotechar)

    def _map_label(self, entity_type):
        if self.label_map is not None:
            return self.label_map[entity_type]
        else:
            return entity_type

    def load_label(self, filename):
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
            if self.label_map is not None:
                labels = [
                    Label(
                        entity_type=self.label_map[row[idx[0]]],
                        start=int(row[idx[1]]),
                        length=int(row[idx[2]]) - int(row[idx[1]]),
                        entity=row[idx[3]]
                    ) for row in csvreader
                ]
            else:
                labels = [
                    Label(
                        entity_type=row[idx[0]],
                        start=int(row[idx[1]]),
                        length=int(row[idx[2]]) - int(row[idx[1]]),
                        entity=row[idx[3]]
                    ) for row in csvreader
                ]

        return labels

    def load_data(self, path) -> dict:
        """Creates a dict the dataset."""
        examples = {'guid': [], 'text': [], 'ann': []}

        # for deid datasets, "path" is a folder containing txt/ann subfolders
        # "txt" subfolder has text files with the text of the examples
        # "ann" subfolder has annotation files with the labels
        txt_path = path / 'txt'
        ann_path = path / 'ann'
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
            labels = self.load_label(ann_path / f'{f[:-4]}.gs')

            examples['guid'].append(guid)
            examples['text'].append(text)
            examples['ann'].append(labels)

        return examples


class DeidDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def get_example(self, i, id2label):
        """Output a tuple for the given index."""
        input_ids = dataset[i]['input_ids'].tolist()
        attention_mask = dataset[i]['attention_mask'].tolist()
        token_type_ids = dataset[i]['token_type_ids'].tolist()
        label_ids = dataset[i]['labels'].tolist()
        labels = ['-100' if l == -100 else id2label[l] for l in label_ids]
        return input_ids, attention_mask, token_type_ids, label_ids, labels
