import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = pd.read_csv("training_results_no_market_cap/epoch_results.csv")
print(data)

# Assuming you want to plot these as an example of working with the data
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
stop_index = None

# Plot training and validation loss per epoch
axs[0].plot(data['epoch'][:stop_index], data['train_loss'][:stop_index], label='Train Loss')
axs[0].plot(data['epoch'][:stop_index], data['val_loss'][:stop_index], label='Validation Loss')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot training and validation accuracy per epoch
axs[1].plot(data['epoch'][:stop_index], data['train_accuracy'][:stop_index], label='Train Accuracy')
axs[1].plot(data['epoch'][:stop_index], data['val_accuracy'][:stop_index], label='Validation Accuracy')
axs[1].set_title('Training and Validation Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

# Plot training and validation F1 score per epoch
axs[2].plot(data['epoch'][:stop_index], data['train_f1'][:stop_index], label='Train F1 Score')
axs[2].plot(data['epoch'][:stop_index], data['val_f1'][:stop_index], label='Validation F1 Score')
axs[2].set_title('Training and Validation F1 Score')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('F1 Score')
axs[2].legend()

plt.tight_layout()
plt.savefig("model_performance.png")

