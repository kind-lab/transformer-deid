import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizerFast
from transformer_deid.tokenization import split_sequences

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_logits(encodings, model):
    """ Return predicted labels from the encodings of a *single* text example. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = model(input_ids=torch.tensor(encodings['input_ids']).to(device),
                attention_mask=torch.tensor(encodings['attention_mask']).to(device))
    logits = result['logits'].cpu().detach().numpy()
    return logits


def deid_example(text, model):
    """ Run deid on a single instance of text input. Return replaced text. """
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    texts = split_sequences(tokenizer, [text])
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
    result = replace_names(encodings.tokens, pred_labels, label_id=6, repl='___')
    return result


def replace_names(tokens, labels, label_id, repl='___'):   # TODO: combine tokens into words
    """ Replace predicted name tokens with repl. """
    tokens = list(tokens)
    for index, label in enumerate(labels):
        if label == label_id:
            tokens[index] = repl
    return ' '.join(tokens)
