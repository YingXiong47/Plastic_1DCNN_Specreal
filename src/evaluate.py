import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Spectral1DCNN
from utils import SpectralDataset, load_processed_json


DATA_PATH = "data/processed/processed_4channel_spectral_dataset.json"
MODEL_PATH = "models/plastic_1dcnn_model.pt"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

BATCH_SIZE = 16


def predict(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = softmax(logits)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate_loss_accuracy(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_to_int, int_to_label = load_processed_json(DATA_PATH)

    test_dataset = SpectralDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    num_classes = len(label_to_int)

    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = Spectral1DCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate_loss_accuracy(
        model,
        test_loader,
        criterion,
        device
    )

    y_pred, y_true, y_probs = predict(
        model,
        test_loader,
        device
    )

    class_names = [int_to_label[str(i)] for i in range(num_classes)]

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_acc}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Evaluation complete.")


if __name__ == "__main__":
    main()