import torch.nn as nn
from tqdm import tqdm
from model import BertForSequenceRegression
from transformers import AdamW
import sys
import torch
from sklearn.metrics import f1_score, fbeta_score

def train(model: BertForSequenceRegression,
        train_dataloader,
        optimizer: AdamW,
        device,
        epoch) -> BertForSequenceRegression:

    ce_loss = nn.BCELoss()

    total_loss = 0
    sum_corect = 0
    num_ques = 0
    prediction = []
    labels = []

    train_bar = tqdm(train_dataloader)

    for batch_idx, batch in enumerate(train_bar):
        batch_size = len(batch['input_ids'])
        batch = {key: value.to(device) for key, value in batch.items()}

        # forward
        output = model(input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'])

        loss = ce_loss(output.float(), batch['label'].float())
        preds = torch.where(output > 0.5, 1, 0)
        prediction.extend(preds.to('cpu'))
        labels.extend(batch['label'].to('cpu'))
        num_ques += preds.shape[0]
        sum_corect += (preds == batch['label']).sum().item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

        train_bar.set_description(f"epoch : {epoch+1} train_loss : {round(total_loss / (batch_idx + 1), 3):.4f} train_acc : {sum_corect / num_ques:.4f}")

    total_loss = total_loss / len(train_dataloader.dataset)
    score = fbeta_score(labels, prediction, beta=7)
    print(f'f1_score={score:.3f}')
    print(f'train_loss={total_loss:.3f}')

    return model 