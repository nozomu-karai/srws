from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset import SRWS_test, SRWS
from torch.utils.data import DataLoader, random_split
from utils import set_seed, set_device
from model import BertForSequenceRegression
from train import train
from evaluate import evaluate
import torch
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score, fbeta_score
import numpy as np
from argparse import ArgumentParser
from pathlib import Path



def main():
    set_seed(0)
    device = set_device("0,1,2")
    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    dataset = SRWS_test(path="data/test.csv", tokenizer=tokenizer, max_seq_len=512)
    batch_size = 32
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataset = SRWS(path="data/train.csv", tokenizer=tokenizer, max_seq_len=512)
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = BertForSequenceRegression().to(device)
    state_dict = torch.load("scibert/model.pth", map_location=device)
    model.load_state_dict(state_dict)
    prediction, labels, outputs = [], [], []

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)


    model.eval()
    with torch.no_grad():
        val_bar = tqdm(val_dataloader)
        dev_bar = tqdm(test_dataloader)
        for batch_idx, batch in enumerate(val_bar):
            batch_size = len(batch['input_ids'])
            batch = {key: value.to(device) for key, value in batch.items()}

            # forward
            output = model(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'])

            labels.extend(batch['label'].to('cpu'))
            outputs.extend(output.to('cpu'))

        best_score = -1
        best_c = 0
        for c in np.arange(0, 1, 0.05):
            preds = np.where(outputs > c, 1, 0)
            score = fbeta_score(labels, preds, beta=7)
            if score > best_score:
                best_score = score
                best_c = c
        
        print(f"fbeta_score : {best_score}")

        for batch_idx, batch in enumerate(dev_bar):
            batch_size = len(batch['input_ids'])
            batch = {key: value.to(device) for key, value in batch.items()}

            # forward
            output = model(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'])

            preds = torch.where(output > best_c, 1, 0)
            prediction.extend(preds.to('cpu'))
    csv_file = open('data/sample_submit.csv', "r")
    f = csv.reader(csv_file, delimiter=",")
    num = []
    for idx, low in enumerate(f):
        num.append(low[0])
    with open("scibert/output_best.csv", "wt") as f:
        for n, pre in zip(num, prediction):
            f.write(f"{n},{pre}\n")
    

if __name__ == '__main__':
    main() 