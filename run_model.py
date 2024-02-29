import torch
from torch import nn
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
import numpy as np
import pickle

def get_loss_function(model):
    if model.is_regression:
        return nn.MSELoss()
    else:
        return nn.CrossEntropyLoss()


def check_gradients_for_nan(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:  # Parameters might not have gradients if they are not trainable or didn't participate in the forward pass
            if torch.isnan(parameter.grad).any():
                print(f"NaN gradient found in {name}")
                return True  # Indicate that a NaN gradient was found
    return False  # No NaN gradients were found


def train_model(epoch, model, loader, optimizer, device):

    model.train()
    loss_fn = get_loss_function(model)
    total_loss = 0

    for batch_num, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        financial_data = batch['numerical'].to(device)

        labels = batch['labels'].to(device)

        # Handle regression and classification label format
        if not model.is_regression:
            labels = labels.long()  # Ensure labels are long type for classification

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)

        # For regression, ensure the outputs and labels have the same dimensions
        if model.is_regression:
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
            raise ValueError("NaN gradient detected, stopping training")
        else:
            optimizer.step()

        optimizer.step()  # Proceed with optimizer step if no NaN gradients were detected
        print(f"LOSS: {loss.item()}, batch {batch_num} / {len(loader)}")

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(loader)}")


def evaluate_model(model, loader, device):
    model.eval()
    loss_fn = get_loss_function(model)
    total_loss = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch_num, batch in enumerate(loader):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            financial_data = batch['numerical'].to(device)    
            labels = batch['labels'].to(device)

            # Handle regression and classification label format
            if not model.is_regression:
                labels = labels.long()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)
            
            # For regression, ensure the outputs and labels have the same dimensions
            if model.is_regression:
                outputs = outputs.squeeze(-1)
                labels = labels.float()

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            if model.is_regression:
                all_predictions.extend(outputs.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
            else:
                _, predicted_labels = torch.max(outputs, 1)
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

    metrics = {}
    metrics['loss'] = total_loss / len(loader)
    
    if model.is_regression:
        mse = mean_squared_error(all_true_labels, all_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_true_labels, all_predictions)
        metrics['rmse'] = rmse
        metrics['r2'] = r2
        print(f"Loss: {metrics['loss']}, RMSE: {rmse}, R^2: {r2}")
    else:
        accuracy = accuracy_score(all_true_labels, all_predictions)
        f1 = f1_score(all_true_labels, all_predictions, average='weighted')
        metrics['accuracy'] = accuracy
        metrics['f1_score'] = f1
        print(f"Loss: {metrics['loss']}, Accuracy: {accuracy}, F1 Score: {f1}")

    return metrics
