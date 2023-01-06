import csv
import os
from tqdm import tqdm
from importlib.resources import open_text
import json
from transformer_deid.label import Label
from transformer_deid.data import DeidDataset
from transformer_deid.tokenization import split_sequences, assign_tags, encode_tags


def load_label(filename):
    """
    Loads annotations from a CSV file.
    CSV file should have entity_type/start/stop columns.
    """
    # can change this, make it a variable? options: "base", "hipaa", "binary"
    label_map = get_label_map('base')

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


def load_data(path) -> dict:
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
        if os.path.isdir(ann_path):
            labels = load_label(ann_path / f'{f[:-4]}.gs')
            examples['ann'].append(labels)

        examples['guid'].append(guid)
        examples['text'].append(text)

    return examples


def create_deid_dataset(data_dict, tokenizer, label2id=None) -> DeidDataset:
    """Creates a dataset from set of texts and labels.

       Args:
            data_dict: dict of text data with 'text', 'ann', and 'guid' keys; e.g., from load_data()
            tokenizer: HuggingFace tokenizer, e.g., loaded from AutoTokenizer.from_pretrained()
            label2id: dict to convert label (e.g., 'O,' 'DATE') to id (e.g., 0, 1)

       Returns: DeidDataset; see class definition in data.py
    """

    # specify dataset arguments
    split_long_sequences = True

    # split text/labels into multiple examples
    # (1) tokenize text
    # (2) identify split points
    # (3) output text as it was originally
    texts = data_dict['text']
    labels = data_dict['ann']
    ids = data_dict['guid']

    if split_long_sequences:
        split_dict = split_sequences(tokenizer, texts, labels, ids=ids)

    texts = split_dict['texts']
    labels = split_dict['labels']
    guids = split_dict['guids']

    encodings = tokenizer(texts,
                          is_split_into_words=False,
                          return_offsets_mapping=True,
                          padding=True,
                          truncation=True)

    if labels != []:
        # use the offset mappings in train_encodings to assign labels to tokens
        tags = assign_tags(encodings, labels)

        # encodings are dicts with three elements:
        #   'input_ids', 'attention_mask', 'offset_mapping'
        # these are used as kwargs to model training later
        tags = encode_tags(tags, encodings, label2id)

    else:
        tags = None

    # prepare a dataset compatible with Trainer module
    encodings.pop("offset_mapping")
    dataset = DeidDataset(encodings, tags, guids)

    return dataset


def get_label_map(transform):
    """ Gets dictionary of labels to convert from specific labels (may differ across datasets) to more general labels. """
    # TODO: fix this. open_text is going to kill me.
    # with open_text('transformer_deid', 'label.json') as fp:
    with open('transformer_deid/label.json') as fp:
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


def get_labels(labels_each_doc):
    """Gets the list of labels for this data set."""
    unique_labels = list(
        set(label.entity_type for labels in labels_each_doc
            for label in labels))

    unique_labels.sort()

    # add in the object tag - ensure it is the first element for label2id and id2label dict
    if 'O' in unique_labels:
        unique_labels.pop('O')
    unique_labels = ['O'] + unique_labels

    return unique_labels


def save_labels(labels, ids, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    header = ['start', 'stop', 'entity', 'entity_type']

    for doc, id in tqdm(zip(labels, ids), total=len(ids)):
        label_list = [[
            label.start, label.start + label.length, label.entity,
            label.entity_type
        ] for label in doc]
        with open(f'{out_path}/{id}.gs', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(label_list)
