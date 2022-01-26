from transformer_deid import tokenization


def test_assign_tags(tokenizer):
    """Verify the assign_tags function assigns labels correctly to a sequence."""
    text = 'Hello my name is Alistair Johnson and I do not have a pneumothorax.'
    start, length = 17, 16
    name_tokens = ['Alistair', 'Johnson']

    labels = [
        [
            tokenization.Label(
                start=start,
                length=length,
                entity_type='NAME',
                entity=text[start:start + length]
            )
        ]
    ]

    encodings = tokenizer(
        [text],
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )

    tags = tokenization.assign_tags(encodings, labels)

    # only care about the first (and only) sequence
    tags = tags[0]
    encoding = encodings[0]
    tokens = tokenizer.convert_ids_to_tokens(encoding.ids)

    for n in name_tokens:
        idx = tokens.index(n)
        assert tags[idx] == 'NAME'


def test_split_sequence_short(tokenizer):
    """ Verify sequence splitting according to max sequence length is done correctly. """
    text = 'Hello my name is Alistair Johnson and I do not have a pneumothorax.'
    # Case for short sentence
    assert tokenization.split_sequences(tokenizer, [text]) == [text]


def test_split_sequence_long(tokenizer):
    """ Verify sequence splitting according to max sequence length is done correctly. """
    long_text = 'Hello my name is Alistair Johnson and I do not have a pneumothorax. ' * 100
    split_long = tokenization.split_sequences(tokenizer, [long_text])

    # Case for single test sentence
    for split_text in split_long[:-2]:
        assert len(tokenizer(split_text, add_special_tokens=False)['input_ids']) == tokenizer.max_len_single_sentence
    assert len(tokenizer(split_long[:-1], add_special_tokens=False)['input_ids']) <= tokenizer.max_len_single_sentence

    assert ''.join(split_long) == long_text


def test_split_sequence_long_batch(tokenizer):
    """ Verify sequence splitting according to max sequence length is done correctly. """
    long_text = 'Hello my name is Alistair Johnson and I do not have a pneumothorax. ' * 100
    split_long = tokenization.split_sequences(tokenizer, [long_text, long_text])

    # Case for multiple test sentences
    half = int(len(split_long) / 2)
    # First sentence
    for split_text in split_long[: half - 2]:
        assert len(tokenizer(split_text, add_special_tokens=False)['input_ids']) == tokenizer.max_len_single_sentence
    assert len(tokenizer(split_long[half - 1], add_special_tokens=False)['input_ids']) <= tokenizer.max_len_single_sentence
    # Second sentence
    for split_text in split_long[half: half - 2]:
        assert len(tokenizer(split_text, add_special_tokens=False)['input_ids']) == tokenizer.max_len_single_sentence
    assert len(tokenizer(split_long[-1], add_special_tokens=False)['input_ids']) <= tokenizer.max_len_single_sentence

    assert ''.join(split_long) == long_text*2