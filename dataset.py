import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch

class SRWS(Dataset):
    def __init__(self, path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.inputs, self.labels = self.load(path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        outputs = self.tokenizer(self.inputs[idx][0], self.inputs[idx][1],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_seq_len,
                                return_token_type_ids=True,
                                padding="max_length",
                                return_attention_mask=True,
                                return_tensors="pt",
                                return_length=True)
        input_ids = outputs['input_ids'].reshape(self.max_seq_len)
        attention_mask = outputs['attention_mask'].reshape(self.max_seq_len) # 0 if pad else 1
        token_type_ids = outputs['token_type_ids'].reshape(self.max_seq_len) # 1 if target segment else 0

        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'label': torch.tensor(int(self.labels[idx]))
        }
        

    def load(self, path):
        inputs, labels = [], []
        csv_file = open(path, "r")
        f = csv.reader(csv_file, delimiter=",")
        for idx, row in enumerate(f):
            if idx == 0:
                continue

            inputs.append([row[1], row[2]])
            labels.append(row[3])

        assert len(inputs) == len(labels)

        return inputs, labels


class SRWS_test(Dataset):
    def __init__(self, path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.inputs = self.load(path)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        outputs = self.tokenizer(self.inputs[idx][0], self.inputs[idx][1],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_seq_len,
                                return_token_type_ids=True,
                                padding="max_length",
                                return_attention_mask=True,
                                return_tensors="pt",
                                return_length=True)
        input_ids = outputs['input_ids'].reshape(self.max_seq_len)
        attention_mask = outputs['attention_mask'].reshape(self.max_seq_len) # 0 if pad else 1
        token_type_ids = outputs['token_type_ids'].reshape(self.max_seq_len) # 1 if target segment else 0

        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        }
        

    def load(self, path):
        inputs = []
        csv_file = open(path, "r")
        f = csv.reader(csv_file, delimiter=",")
        for idx, row in enumerate(f):
            if idx == 0:
                continue

            inputs.append([row[1], row[2]])

        return inputs