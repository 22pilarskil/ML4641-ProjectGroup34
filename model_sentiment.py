import torch
from torch import nn
from transformers import BertModel

class BertForSentimentAnalysis(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=3):

        super(BertForSentimentAnalysis, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # Shape: (batch_size, num_labels)
        return logits

