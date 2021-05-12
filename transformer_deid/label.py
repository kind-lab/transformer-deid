"""Classes and functions for working with entity labels."""

class Label(object):
    """Base class for a label.

    A label contains four attributes of primary interest:
        * entity_type - the type of entity which is labeled
        * start - the start offset of the label in the source text
        * length - the length of the label in the source text
        * entity - the actual text of the entity
    """
    def __init__(self, entity_type, start, length, entity):
        """Initialize a data processor with the location of the data."""
        self.entity_type = entity_type
        self.start = start
        self.length = length
        self.entity = entity

    def __repr__(self):
        return f'Label({self.entity_type}, {self.start}, {self.length}, {self.entity})'

    def map_entity_type(self, mapping, force_upper=True):
        if force_upper:
            self.entity_type = mapping[self.entity_type.upper()]
        else:
            self.entity_type = mapping[self.entity_type]

    def shift(self, s):
        self.start += s
        return self

    def contains(self, i):
        """Returns true if any label contains the offset."""
        return (self.start >= i) & ((self.start + self.length) < i)

    def overlaps(self, start, stop):
        """Returns true if any label contains the start/stop offset."""
        contains_start = (self.start >= start) & (self.start < stop)
        contains_stop = ((self.start + self.length) >=
                         start) & ((self.start + self.length) < stop)
        return contains_start | contains_stop
    
    def within(self, start, stop):
        """Returns true if the label is within a start/stop offset."""
        after_start = (self.start >= start) or ((self.start + self.length) >= start)
        before_stop = self.start < stop
        return after_start & before_stop


def convert_to_bio_scheme(tokens: list) -> list:
    def b_or_i(w, w_prev):
        if w == 'O':
            return 'O'
        elif w == w_prev:
            return f'I-{w}'
        else:
            return f'B-{w}'

    return [
        [b_or_i(w, None if i == 0 else sequence[i-1]) for i, w in enumerate(sequence)]
        for sequence in tokens
    ]
