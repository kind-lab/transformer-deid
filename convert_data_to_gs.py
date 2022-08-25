# reformats various datasets for de-identification into our format
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert i2b2 annotations')
parser.add_argument(
    '-d',
    '--data_type',
    type=str,
    default=None,
    required=True,
    choices=[
        'i2b2_2006', 'i2b2_2014', 'physionet', 'physionet_google', 'opendeid'
    ],
    help='source dataset (impacts processing)'
)
parser.add_argument(
    '-i',
    '--input',
    type=str,
    default=None,
    required=True,
    help='folder or file to convert'
)
parser.add_argument(
    '-o',
    '--output',
    type=str,
    default=None,
    required=True,
    help='folder to output converted annotations'
)

# optionally also output a single CSV with all the data
parser.add_argument(
    '-q',
    '--quiet',
    action='store_true',
    help='suppress peasants discussing their work'
)

# define a dictionary of constant values for each dataset
i2b2_2014 = {'tag_list': ['id', 'start', 'end', 'text', 'TYPE', 'comment']}

physionet_gs = {
    'columns':
        ['patient_id', 'record_id', 'start', 'stop', 'entity_type', 'entity']
}

physionet_google = {'columns': ['record_id', 'begin', 'length', 'type']}

opendeid = {'tag_list': ['id', 'start', 'end', 'text', 'TYPE']}

# our output dataframe will have consistent columns
COLUMN_NAMES = [
    'document_id', 'annotation_id', 'start', 'stop', 'entity', 'entity_type',
    'comment'
]


def load_physionet_text(text_filename, verbose_flag=False):
    """Loads text from the PhysioNet id.text file.
    
    Output
    reports - list with each element being the text of a single record
    document_ids - list with the document_id for the record
    """
    reports, document_ids = [], []

    with open(text_filename, 'r') as fp:
        END_OF_RECORD = True
        reader = fp.readlines()
        if verbose_flag:
            reader = tqdm(reader)
        for line in reader:
            if END_OF_RECORD:
                # skip empty rows
                if line == '\n':
                    continue

                # make sure this is the start of a new record
                if line[0:16] != 'START_OF_RECORD=':
                    raise ValueError(
                        'Record ended, but "START_OF_RECORD" not found in next line.'
                    )
                line = line[16:].split('||||')
                # last element will be newline, so we ignore it
                text = []
                pt_id = line[0]
                doc_id = line[1]
                END_OF_RECORD = False
                continue

            if line == '||||END_OF_RECORD\n':
                END_OF_RECORD = True
                reports.append(''.join(text))

                document_id = pt_id + '-' + doc_id
                document_ids.append(document_id)
                continue

            text.append(line)

    return reports, document_ids


def load_physionet_gs(input_path, verbose_flag):
    text_filename = os.path.join(input_path, 'id.text')
    ann_filename = os.path.join(input_path, 'id-phi.phrase')

    if verbose_flag:
        print(f'Loading text from {text_filename}')

    # read in text into list of lists
    # each sublist has:
    #   patient id, record id, text
    reports, document_ids = load_physionet_text(
        text_filename, verbose_flag=verbose_flag
    )

    if verbose_flag:
        print(f'Loading annotations from {ann_filename}')

    # load in PHI annotations
    annotations = []
    with open(ann_filename, 'r') as fp:
        reader = fp.readlines()
        if verbose_flag:
            reader = tqdm(reader)
        for line in reader:
            annot = line[0:-1].split(' ')
            # reconstitute final entity as it may have a space
            annotations.append(annot[0:5] + [' '.join(annot[5:])])

    # convert annotations to dataframe
    df = pd.DataFrame(annotations, columns=physionet_gs['columns'])

    # unique document identifier is 'pt_id-rec_id'
    df['document_id'] = df['patient_id'] + '-' + df['record_id']
    df.drop(['patient_id', 'record_id'], axis=1, inplace=True)
    df['start'] = df['start'].astype(int)
    df['stop'] = df['stop'].astype(int)

    # create other columns for needed output fields
    df.sort_values(['document_id', 'start', 'stop'], inplace=True)
    df['annotation_id'] = df.groupby('document_id').cumcount() + 1
    df['comment'] = None

    return reports, df, document_ids


def load_physionet_google(input_path, verbose_flag):
    text_filename = os.path.join(input_path, 'id.text')
    ann_filename = os.path.join(
        input_path, 'I2B2-2014-Relabeled-PhysionetGoldCorpus.csv'
    )

    if verbose_flag:
        print(f'Loading text from {text_filename}')

    # read in text into list of lists
    # each sublist has:
    #   patient id, record id, text
    reports, document_ids = load_physionet_text(
        text_filename, verbose_flag=verbose_flag
    )

    if verbose_flag:
        print(f'Loading annotations from {ann_filename}')

    # load in PHI annotations
    df = pd.read_csv(ann_filename, header=0, sep=',')

    # unique document identifier is 'pt_id-rec_id'
    df['document_id'] = df['record_id'].apply(
        lambda x: '-'.join(x.split('||||')[:2])
    )

    df['start'] = df['begin'].astype(int)
    df['stop'] = df['start'] + df['length'].astype(int)

    df.drop(['record_id', 'begin', 'length'], axis=1, inplace=True)
    df.rename(columns={'type': 'entity_type'}, inplace=True)

    # create other columns for needed output fields
    df.sort_values(['document_id', 'start', 'stop'], inplace=True)
    df['annotation_id'] = df.groupby('document_id').cumcount() + 1
    df['comment'] = None

    # add the entity to the annotation dataframe
    entities = []
    for i, row in df.iterrows():
        idx = document_ids.index(row['document_id'])
        entities.append(reports[idx][row['start']:row['stop']])

    df['entity'] = entities

    return reports, df, document_ids


def load_opendeid(input_path, verbose_flag):
    """
       Args: 
            input_path: 
            verbose_flag:

       Returns:
            reports:
            annotations:
            document_ids:
    """
    files = os.listdir(input_path)

    # filter to files of a given extension
    files = [f for f in files if f.endswith('.xml')]

    if len(files) == 0:
        print(f'No files found in folder {input_path}')
        return None, None, None

    if verbose_flag:
        N = len(files)
        print(f'Processing {N} files found in {input_path}')
        files = tqdm(files)

    records, annotations, document_ids = [], [], []

    for f in files:
        # document ID is filename minus last extension
        document_id = f.split('.')
        if len(document_id) > 1:
            document_id = '.'.join(document_id[0:-1])
        else:
            document_id = document_id[0]

        # load as XML tree
        fn = os.path.join(input_path, f)
        with open(fn, 'r', encoding='UTF-8') as fp:
            xml_data = fp.read()

        tree = ET.fromstring(xml_data)

        # get the text from TEXT field
        text = tree.find('TEXT')
        if text is not None:
            text = text.text
        else:
            print(f'WARNING: {fn} did not have any text.')

        # the <TAGS> section has deid annotations
        tags_xml = tree.find('TAGS')

        # example tag:
        # <DATE id="P0" start="16" end="20" text="2069" TYPE="DATE" comment="" />
        # <ID id="I0" text="123A1231231" TYPE="IDNUM" start="14" end="24"/>
        tags = list()
        for tag in tags_xml:
            tags.append(
                [document_id] + [tag.get(t)
                                 for t in opendeid['tag_list']] + ['']
            )

        records.append(text)
        annotations.extend(tags)
        document_ids.append(document_id)

    # convert annotations to dataframe
    annotations = pd.DataFrame(annotations, columns=COLUMN_NAMES)
    annotations['start'] = annotations['start'].astype(int)
    annotations['stop'] = annotations['stop'].astype(int)

    return records, annotations, document_ids


def load_i2b2_2014(input_path, verbose_flag):
    files = os.listdir(input_path)

    # filter to files of a given extension
    files = [f for f in files if f.endswith('.xml')]

    if len(files) == 0:
        print(f'No files found in folder {input_path}')
        return None, None, None

    if verbose_flag:
        N = len(files)
        print(f'Processing {N} files found in {input_path}')
        files = tqdm(files)

    records, annotations, document_ids = [], [], []
    for f in files:
        # document ID is filename minus last extension
        document_id = f.split('.')
        if len(document_id) > 1:
            document_id = '.'.join(document_id[0:-1])
        else:
            document_id = document_id[0]

        # load as XML tree
        fn = os.path.join(input_path, f)
        with open(fn, 'r', encoding='UTF-8') as fp:
            xml_data = fp.read()

        tree = ET.fromstring(xml_data)

        # get the text from TEXT field
        text = tree.find('TEXT')
        if text is not None:
            text = text.text
        else:
            print(f'WARNING: {fn} did not have any text.')

        # the <TAGS> section has deid annotations
        tags_xml = tree.find('TAGS')

        # example tag:
        # <DATE id="P0" start="16" end="20" text="2069" TYPE="DATE" comment="" />
        tags = list()
        for tag in tags_xml:
            tags.append(
                [document_id] + [tag.get(t) for t in i2b2_2014['tag_list']]
            )

        records.append(text)
        annotations.extend(tags)
        document_ids.append(document_id)

    # convert annotations to dataframe
    annotations = pd.DataFrame(annotations, columns=COLUMN_NAMES)
    annotations['start'] = annotations['start'].astype(int)
    annotations['stop'] = annotations['stop'].astype(int)

    return records, annotations, document_ids


def load_i2b2_2006(input_path, verbose_flag):
    # input_path should be the name of a file
    # text_filename = os.path.join(input_path, 'id.text')
    # ann_filename = os.path.join(input_path, 'id-phi.phrase')
    # e.g. i2b2_2006/deid_surrogate_test_all_groundtruth_version2.xml

    # load as XML tree
    with open(input_path, 'r', encoding='UTF-8') as fp:
        xml_data = fp.read()

    tree = ET.fromstring(xml_data)

    reader = tree.iter('RECORD')
    if verbose_flag:
        N = len(tree.findall('RECORD'))
        print(f'Processing {N} records found in {input_path}')
        reader = tqdm(tree.iter('RECORD'), total=N)

    records, ann, document_ids = [], [], []
    # get the text from TEXT field
    for record in reader:
        document_id = record.get('ID')

        # the <TEXT> element contains text like so:
        # <TEXT>This is a note, with <PHI TYPE="NAME">Peter's</PHI> name</TEXT>
        # need to iterate through the elements and track offsets
        # also build the text string along the way
        text_tag = record.find('TEXT')
        n = 0
        # initialize with text in the text tag
        text = text_tag.text
        n += len(text)
        ann_id = 0
        for t in list(text_tag):
            if t.tag == 'PHI':
                start = n
                stop = n + len(t.text)
                ann.append(
                    [document_id, ann_id, start, stop, t.text,
                     t.get('TYPE')]
                )

            text += t.text
            n += len(t.text)
            text += t.tail
            n += len(t.tail)

        records.append(text)
        document_ids.append(document_id)

    # convert annotations to dataframe
    ann = pd.DataFrame(
        ann,
        columns=[
            'document_id', 'annotation_id', 'start', 'stop', 'entity',
            'entity_type'
        ]
    )
    ann['start'] = ann['start'].astype(int)
    ann['stop'] = ann['stop'].astype(int)
    ann['comment'] = None

    return records, ann, document_ids


def get_data_type_info(data_type):
    if data_type == 'i2b2_2014':
        return load_i2b2_2014
    elif data_type == 'physionet':
        return load_physionet_gs
    elif data_type == 'physionet_google':
        return load_physionet_google
    elif data_type == 'i2b2_2006':
        return load_i2b2_2006
    elif data_type == 'opendeid':
        return load_opendeid
    else:
        raise ValueError(f'Unrecognized: --data {data_type}')


def main(args):
    args = parser.parse_args(args)

    input_path = args.input
    out_path = args.output
    verbose_flag = not args.quiet

    # prep output folders if they don't exist
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(os.path.join(out_path, 'ann')):
        os.mkdir(os.path.join(out_path, 'ann'))
    if not os.path.exists(os.path.join(out_path, 'txt')):
        os.mkdir(os.path.join(out_path, 'txt'))

    load_dataset = get_data_type_info(args.data_type)

    reports, annotations, document_ids = load_dataset(input_path, verbose_flag)

    if document_ids is None:
        # no data was loaded
        return

    if verbose_flag:
        print('Writing out annotation and text files.')
        document_ids = tqdm(document_ids)

    # loop through reports to output files
    for i, document_id in enumerate(document_ids):
        idx = annotations['document_id'] == document_id
        df_out = annotations.loc[idx, COLUMN_NAMES]

        # output dataframe style PHI
        df_out.to_csv(
            os.path.join(out_path, 'ann', document_id + '.gs'), index=False
        )
        with open(
            os.path.join(out_path, 'txt', document_id + '.txt'), 'w'
        ) as fp:
            fp.write(reports[i])

    if verbose_flag:
        i += 1
        print(f'Success!')
        print(f'Output {i} files to {out_path}{os.sep}ann')
        print(f'   and {i} files to {out_path}{os.sep}txt')


if __name__ == '__main__':
    main(sys.argv[1:])
