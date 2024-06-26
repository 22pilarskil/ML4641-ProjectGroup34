import torch
from torch import nn
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score
import numpy as np
import pickle
import csv

def get_loss_function(model):
    if model.module.is_regression:
        return nn.MSELoss()
    else:
        return nn.CrossEntropyLoss()


def check_gradients_for_nan(model):
    for name, parameter in model.module.named_parameters():
        if parameter.grad is not None:  # Parameters might not have gradients if they are not trainable or didn't participate in the forward pass
            if torch.isnan(parameter.grad).any():
                print(f"NaN gradient found in {name}")
                return True  # Indicate that a NaN gradient was found
    return False  # No NaN gradients were found


class RunningAverage:

    def __init__(self):
        self.total_sum = 0  # To store the sum of all numbers added
        self.count = 0      # To count the number of elements

    def update(self, number):
        self.total_sum += number
        self.count += 1

    def get_average(self):
        if self.count == 0:
            return 0  # To avoid division by zero
        return self.total_sum / self.count


def train_model(epoch, model, train_loader, val_loader, optimizer, device, iteration_results_file):

    print("TRAINING")
    model.train()
    loss_fn = get_loss_function(model)
    total_loss = 0
    average_loss = RunningAverage()

    for batch_num, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        financial_data = batch['numerical'].to(device)

        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)
        _, preds = torch.max(outputs, 1)
        # For regression, ensure the outputs and labels have the same dimensions
        if model.module.is_regression:
            outputs = outputs.squeeze(-1)
            labels = labels.float()  # Ensure labels are float type for regression

        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        if check_gradients_for_nan(model):
            with open("financial_data.pkl", "wb") as f:
                pickle.dump(financial_data, f)
            print(outputs)
            print(torch.max(financial_data), torch.min(financial_data))
            continue
        else:
            average_loss.update(loss.item())
            optimizer.step()

        print(f"LOSS: {loss.item()}, AVERAGE: {average_loss.get_average()}, batch {batch_num} / {len(train_loader)}, \npreds: {preds}")

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")


def evaluate_model(model, loader, device):

    print("EVALUATING")
    model.eval()

    loss_fn = get_loss_function(model)
    if model.module.is_regression:
        total_loss = 0
        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for batch_num, batch in enumerate(loader):
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                financial_data = batch['numerical'].to(device)    
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)
                
                outputs = outputs.squeeze(-1)
                labels = labels.float()

                loss = loss_fn(outputs, labels)
                total_loss += loss.item()

                all_predictions.extend(outputs.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                print("BATCH {batch_num} / {len(loader)}")

        metrics = {}
        metrics['loss'] = total_loss / len(loader)
        
        mse = mean_squared_error(all_true_labels, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_true_labels, all_predictions)
        metrics['rmse'] = rmse
        metrics['r2'] = r2
        print(f"Loss: {metrics['loss']}, RMSE: {rmse}, R^2: {r2}")
    else:
        metrics = {}

        num_correct = 0
        num_samples = 0
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_num, batch in enumerate(loader):

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                financial_data = batch['numerical'].to(device)    
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask, financial_data)
                loss = loss_fn(logits, labels)  # Compute the loss
                total_loss += loss.item()

                _, preds = torch.max(logits, 1)
                num_correct += (preds == labels).sum().item()
                num_samples += preds.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                print(f"BATCH {batch_num} / {len(loader)} \npreds: {preds}")

        metrics['accuracy'] = num_correct / num_samples
        metrics['f1'] = f1_score(all_labels, all_preds, average='weighted')
        metrics['loss'] = total_loss / len(loader)
        metrics['precision'] = precision_score(all_labels, all_preds, average='weighted')
        metrics['recall'] = recall_score(all_labels, all_preds, average='weighted')


    return metrics
