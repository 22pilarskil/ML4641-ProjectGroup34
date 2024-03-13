import csv
import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer

from model_sentiment import BertForSentimentAnalysis  # Assuming the model class is defined in this module
from datasets import create_data_loaders
from run_sentiment import train_model, evaluate_model

filename = '../liam_test/all-data.csv.csv'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Example tokenizer

batch_size = 32
max_len = 128
train_loader, val_loader, test_loader = create_data_loaders(filename, tokenizer, batch_size, max_len=max_len)

model = BertForSentimentAnalysis(pretrained_model_name='bert-base-uncased', num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 100

results_dir = "./training_results"
os.makedirs(results_dir, exist_ok=True)
weights_dir = os.path.join(results_dir, "weights")
os.makedirs(weights_dir, exist_ok=True)
results_file = os.path.join(results_dir, "epoch_results.csv")

with open(results_file, 'a+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "train_loss", "test_loss", "train_accuracy", "test_accuracy"])

for epoch in range(epochs):
    
    if epoch != 0:

        model.train()
        train_model(epoch, model, train_loader, loss_fn, optimizer, device)

    model.eval()
    train_loss, train_accuracy = evaluate_model(model, train_loader, loss_fn, device)
    test_loss, test_accuracy = evaluate_model(model, test_loader, loss_fn, device)

    weights_path = os.path.join(weights_dir, f"epoch_{epoch + 1}.pickle")
    torch.save(model.state_dict(), weights_path)

    with open(results_file, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, test_loss, train_accuracy, test_accuracy])

