from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset import SRWS
from torch.utils.data import DataLoader, random_split
from utils import set_seed, set_device
from model import BertForSequenceRegression
from train import train
from evaluate import evaluate
import torch
from argparse import ArgumentParser
from pathlib import Path

def main():
    device = set_device("0")
    set_seed(0)
    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    dataset = SRWS(path="data/train.csv", tokenizer=tokenizer, max_seq_len=512)
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = BertForSequenceRegression().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5) 

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    
    best_score = -1
    for epoch in range(3):
        model.train()
        model = train(model, train_dataloader, optimizer, device, epoch)

        model.eval()
        score = evaluate(model, val_dataloader, device)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "scibert/model.pth")
    

if __name__ == "__main__":
    main()