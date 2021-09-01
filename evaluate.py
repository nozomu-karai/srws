import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Tuple
from sklearn.metrics import f1_score, fbeta_score



def evaluate(model,
        dev_data_loader,
        device: torch.device,
        ) -> Tuple[float, List]:

    ce_loss = nn.BCELoss()
    sum_corect, num_ques, total_loss = 0, 0, 0
    prediction = []
    labels = []

    with torch.no_grad():
        dev_bar = tqdm(dev_data_loader)
        for batch_idx, batch in enumerate(dev_bar):
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

            total_loss += loss.item() * batch_size
            
            dev_bar.set_description(f"val_loss : {round(total_loss / (batch_idx + 1), 3):.4f} val_acc : {sum_corect / num_ques:.4f}")

    score = fbeta_score(labels, prediction, beta=7)
    print(f"fbeta_score : {score}")

    return score

