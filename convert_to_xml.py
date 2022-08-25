import os
from lxml import etree as ET
from importlib.resources import open_text
import json
import csv
import argparse


def get_label_map(transform):
    with open_text('transformer_deid', 'label.json') as fp:
        label_map = json.load(fp)

    # label_membership has different label transforms as keys
    if transform not in label_map:
        raise KeyError(
            'Unable to find label transform %s in label.json' % transform
        )
    label_map = label_map[transform]

    # label_map has items "harmonized_label": ["label 1", "label 2", ...]
    # invert this for the final label mapping
    return {
        label: harmonized_label
        for harmonized_label, original_labels in label_map.items()
        for label in original_labels
    }


def text_ann_to_xml(txt_path, ann_path, label_map):
    root = ET.Element('deIdi2b2')

    with open(txt_path, 'r') as fp:
        full_text = ''.join(fp.readlines())
    ET.SubElement(root, 'TEXT').text = full_text

    tags = ET.SubElement(root, 'TAGS')
    with open(ann_path, 'r') as fp:
        csvreader = csv.reader(fp, delimiter=',', quotechar='"')
        header = next(csvreader)
        # identify which columns we want
        idx = [
            header.index('entity_type'),
            header.index('start'),
            header.index('stop'),
            header.index('entity'),
            header.index('comment'),
            header.index('annotation_id')
        ]

        for row in csvreader:
            ET.SubElement(
                tags,
                label_map[row[idx[0]]],
                TYPE=row[idx[0]],
                comment=row[idx[4]],
                end=str(row[idx[2]]),
                id=row[idx[5]],
                start=str(row[idx[1]]),
                text=row[idx[3]]
            )

    tree = ET.ElementTree(root)
    # tree.write('test.xml', encoding='utf8', xml_declaration=True, pretty_print=True)
    return tree


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Convert annotations and text files to .xml readable by pydeid.'
    )

    parser.add_argument(
        '-p',
        '--path',
        type=str,
        help='file containing txt and ann files data to be converted to XML'
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    rootdir = args.path

    label_map = get_label_map('base')

    filepath = os.listdir(rootdir + 'txt')

    os.mkdir(f'{rootdir}/xml/')

    for file in filepath:
        id = file.split('.')[0]
        txt_path = f'{rootdir}txt/{id}.txt'
        ann_path = f'{rootdir}ann/{id}.gs'
        tree = text_ann_to_xml(txt_path, ann_path, label_map)

        outpath = f'{rootdir}xml/{id}.xml'
        tree.write(
            outpath, encoding='utf8', xml_declaration=True, pretty_print=True
        )  #


if __name__ == '__main__':
    main()
