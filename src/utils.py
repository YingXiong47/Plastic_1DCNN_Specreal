import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_processed_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    X_train = np.array(data["train"]["X"], dtype=np.float32)
    y_train = np.array(data["train"]["y"], dtype=np.int64)

    X_val = np.array(data["val"]["X"], dtype=np.float32)
    y_val = np.array(data["val"]["y"], dtype=np.int64)

    X_test = np.array(data["test"]["X"], dtype=np.float32)
    y_test = np.array(data["test"]["y"], dtype=np.int64)

    label_to_int = data["label_to_int"]
    int_to_label = data["int_to_label"]

    return X_train, y_train, X_val, y_val, X_test, y_test, label_to_int, int_to_label


def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total