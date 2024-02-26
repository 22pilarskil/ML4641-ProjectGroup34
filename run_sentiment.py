import torch
from sklearn.metrics import f1_score
import numpy as np

def train_model(epoch, model, loader, loss_fn, optimizer, device):

    total_loss = 0

    for batch_num, batch in enumerate(loader):

        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels = labels.max(dim=1)[1]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(loader)}")



def evaluate_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch_num, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels = labels.max(dim=1)[1]  # Assuming labels are one-hot encoded and you're converting them to class labels

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()

            # Collect all true labels and predictions for F1 score calculation
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    accuracy = correct_predictions / len(loader.dataset)

    # Calculate F1 score
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')

    return total_loss / len(loader), accuracy, f1
