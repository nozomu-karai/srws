import torch
import torch.nn as nn
from transformers import BertModel


class BertForSequenceRegression(nn.Module):
    def __init__(self,
            pretrained_model,
            output_attentions: bool = False) -> None:

        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model, output_attentions=output_attentions)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor):

        # (batch_size, len) -> (batch_size, hidden_dim)
        output = self.bert(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

        cls = output.pooler_output
        # (batch_size, hidden_dim) -> (batch_size, 1)
        output = self.linear(self.dropout(cls)) 
        output = self.sigmoid(output.reshape(output.shape[0]))
        return output