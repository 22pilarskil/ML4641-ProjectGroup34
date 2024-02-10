import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import chardet

class SentimentAnalysisDataset(Dataset):
    def __init__(self, filename, tokenizer, encoding=None, max_len=256):

        if not encoding:
            with open(filename, 'rb') as f:
                result = chardet.detect(f.read())
                encoding = result["encoding"]
            
        self.data = pd.read_csv(filename, encoding=encoding)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 0]
        
        # Encode label as one-hot vector
        label_dict = {"positive": 0, "negative": 1, "neutral": 2}
        label_encoded = label_dict[label]
        label_one_hot = np.eye(len(label_dict))[label_encoded]

        # Tokenize text
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label_one_hot, dtype=torch.float)
        }


def create_data_loaders(filename, tokenizer, batch_size=32, encoding=None, max_len=256):
    # Initialize dataset
    dataset = SentimentAnalysisDataset(filename=filename, tokenizer=tokenizer, encoding=encoding, max_len=max_len)
    
    # Calculate split sizes
    train_size = int(0.8 * len(dataset))
    val_size = test_size = int(0.1 * len(dataset))
    # Adjust train_size to make sure train_size + val_size + test_size == len(dataset)
    train_size += len(dataset) - (train_size + val_size + test_size)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
