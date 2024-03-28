import torch
from torch import nn
from transformers import BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BertForSentimentAnalysis(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=3, num_financial_metrics=7, seq_length=10, hidden_dim=11, is_regression=True):
        super(BertForSentimentAnalysis, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.is_regression = is_regression  # Store the mode of the model

        # Transformer encoder for financial data
        self.financial_transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=num_financial_metrics, nhead=1, batch_first=True, dim_feedforward=hidden_dim),
            num_layers=2
        )
        
        if self.is_regression:
            # The output layer's dimensions depend on whether the model is for regression or classification
            output_dim = 1
            self.output_layer = nn.Linear(768 + num_financial_metrics, output_dim)

            self.dropout = nn.Dropout(0.1)
        else:
            self.classifier = nn.Linear(768 + num_financial_metrics, num_labels)

    def forward(self, input_ids, attention_mask, financial_data):
        # Process text data through BERT

        if torch.isnan(financial_data).any():
            raise ValueError("NaN detected in financial data input")

        if self.is_regression:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = bert_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)

            # Process financial data through Transformer
            financial_outputs = self.financial_transformer(financial_data.float())  # Shape: (batch_size, seq_length, num_financial_metrics)
            cls_financial_output = financial_outputs[:, 0, :]  # Take the first output

            combined_output = torch.cat((cls_output, cls_financial_output), dim=1)
            combined_output = self.dropout(combined_output)

            # Apply the output layer
            output = self.output_layer(combined_output)

            return output
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = bert_outputs.last_hidden_state[:, 0, :]
            
            financial_outputs = self.financial_transformer(financial_data.float())
            cls_financial_output = financial_outputs[:, 0, :]
            
            combined_output = torch.cat((cls_output, cls_financial_output), dim=1)
            
            logits = self.classifier(combined_output)
                
            return logits

