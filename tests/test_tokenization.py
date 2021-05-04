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