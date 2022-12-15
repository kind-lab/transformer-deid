import torch

class DeidDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, ids):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

    def get_example(self, i, id2label):
        """Output a tuple for the given index."""
        input_ids = self.encodings['input_ids'][i]
        attention_mask = self.encodings['attention_mask'][i]
        token_type_ids = self.encodings.encodings[i].type_ids
        label_ids = self.labels[i]
        return input_ids, attention_mask, token_type_ids, label_ids
