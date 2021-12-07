import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def deid_example(text, model):
    """ Run deid on a single instance of text input. Return replaced text. """
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    texts = split_labelless_sequences([text], tokenizer)
    encodings = tokenizer(
        texts,
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )
    encodings.pop("offset_mapping")
    logits = get_logits(encodings, model)
    pred_labels = np.argmax(logits, axis=2)[0]
    result = replace_names(encodings.tokens, pred_labels, repl='___')
    return result


def get_logits(encodings, model):
    """ Return predicted labels from the encodings of a *single* text example. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = model(input_ids=torch.tensor(encodings['input_ids']).to(device),
                attention_mask=torch.tensor(encodings['attention_mask']).to(device))
    logits = result['logits'].cpu().detach().numpy()
    return logits[0]


def replace_names(tokens, labels, repl='___'):   # TODO: combine tokens into words
    """ Replace predicted name tokens with repl. """
    tokens = list(tokens)
    for index, label in enumerate(labels):
        if label == 6:
            tokens[index] = repl
    return ' '.join(tokens)


def split_labelless_sequences(texts, tokenizer):  # TODO: modify original split_sequences()?
    # tokenize the text
    encodings = tokenizer(texts, add_special_tokens=False)
    seq_len = tokenizer.max_len_single_sentence

    # identify the start/stop offsets of the new text
    sequence_offsets = []
    logger.info('Determining offsets for splitting long segments.')
    for i, encoded in tqdm(enumerate(encodings.encodings), total=len(encodings.encodings)):
        offsets = [o[0] for o in encoded.offsets]
        token_sw = [False] + [
            encoded.word_ids[i + 1] == encoded.word_ids[i]
            for i in range(len(encoded.word_ids) - 1)
        ]
        # iterate through text and add create new subsets of the text
        start = 0
        subseq = []
        while start < len(offsets):
            # ensure we do not start on a sub-word token
            while token_sw[start]:
                start -= 1

            stop = start + seq_len
            if stop < len(offsets):
                # ensure we don't split sequences on a sub-word token
                # do this by shortening the current sequence
                while token_sw[stop]:
                    stop -= 1
            else:
                # end the sub sequence at the end of the text
                stop = len(offsets)

            subseq.append(start)

            # update start of next sequence to be end of current one
            start = stop
        
        sequence_offsets.append(subseq)

    new_text = []

    logger.info('Splitting text.')
    for i, subseq in tqdm(enumerate(sequence_offsets), total=len(encodings.encodings)):
        for j, start in enumerate(subseq):
            if j + 1 >= len(subseq):
                stop = len(encodings[i])
            else:
                stop = subseq[j+1]
            
            text_start = encodings[i].offsets[start][0]
            if stop >= len(encodings[i]):
                text_stop = encodings[i].offsets[-1][0] + encodings[i].offsets[-1][1]
            else:
                text_stop = encodings[i].offsets[stop][0]

            # extract the text from the offsets
            new_text.append(texts[i][text_start:text_stop])

    return new_text
