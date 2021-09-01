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
    parser = ArgumentParser()
    parser.add_argument('--save-path', help='path to save models', required=True)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpuid', help='GPU id (supposed to be using only 1 GPU)')
    parser.add_argument('--pretrained-model', default="allenai/scibert_scivocab_uncased", help='pretrained BERT model path')
    #parser.add_argument('--dataset-path', default="data", help='dataset path')
    parser.add_argument('--max-seq-len', default=512, help='max sequence length for BERT input')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', default=1e-5, help='learning rate')
    parser.add_argument('--num-epochs', default=3, help="number of epochs")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    device = set_device(args.gpuid)
    set_seed(args.seed)
    n_gpu = torch.cuda.device_count()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    dataset = SRWS(path="data/train.csv", tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    n_samples = len(dataset)
    train_size = int(n_samples * 0.8)
    val_size = n_samples - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = BertForSequenceRegression(args.pretrained_model).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr) 

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
            torch.save(model.state_dict(), save_path / "model.pth")
    

if __name__ == "__main__":
    main()