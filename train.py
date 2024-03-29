import os
import csv
import torch
from transformers import AdamW
from model_sentiment import BertForSentimentAnalysis
from dataloader import create_data_loaders
from run_model import train_model, evaluate_model

def run_training():

    data_folder = "data/NumericalData_processed/"
    headlines_file = "data/output_dataset.pkl"
    pretrained_model_name = 'bert-base-uncased'
    batch_size = 16
    is_regression = False  # Adjust based on your task

    train_loader, val_loader, test_loader = create_data_loaders(headlines_file, data_folder, trading_days_before=10, 
                                                                trading_days_after=-1, batch_size=batch_size, is_regression=is_regression)

    model = BertForSentimentAnalysis(pretrained_model_name=pretrained_model_name, is_regression=is_regression)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-7)

    epochs = 100  # Adjust as needed
    results_dir = "./training_results"
    os.makedirs(results_dir, exist_ok=True)
    weights_dir = os.path.join(results_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "epoch_results.csv")
    iterations_file = os.path.join(results_dir, "iteration_results.csv")

    header = ["epoch", "train_loss", "val_loss"]
    if is_regression:
        header.extend(["train_rmse", "train_r2", "val_rmse", "val_r2"])
    else:
        header.extend(["train_accuracy", "train_precision", "train_recall", "train_f1", "val_accuracy", "val_precision", "val_recall", "val_f1"])
    
    with open(results_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    with open(iterations_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'batch_num', 'loss', 'avg_loss', 'accuracy', 'precision', 'recall', 'f1'])

    for epoch in range(epochs):
        train_model(epoch, model, train_loader, val_loader, optimizer, device, iterations_file)
        
        train_metrics = evaluate_model(model, train_loader, device)
        val_metrics = evaluate_model(model, val_loader, device)
        
        weights_path = os.path.join(weights_dir, f"epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), weights_path)

        # Log results
        row = [epoch + 1, train_metrics['loss'], val_metrics['loss']]
        if is_regression:
            row.extend([train_metrics['rmse'], train_metrics['r2'], val_metrics['rmse'], val_metrics['r2']])
        else:
            row.extend([train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1'], 
                        val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['f1']])
        
        with open(results_file, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

if __name__ == "__main__":
    run_training()

