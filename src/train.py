import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Spectral1DCNN
from utils import SpectralDataset, load_processed_json, plot_training_history, calculate_accuracy


DATA_PATH = "data/processed/processed_4channel_spectral_dataset.json"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "plastic_1dcnn_model.pt")
RESULTS_DIR = "results"

BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
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

            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading processed dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_to_int, int_to_label = load_processed_json(DATA_PATH)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    num_classes = len(label_to_int)
    print("Number of classes:", num_classes)
    print("Classes:", label_to_int)

    train_dataset = SpectralDataset(X_train, y_train)
    val_dataset = SpectralDataset(X_val, y_val)
    test_dataset = SpectralDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = Spectral1DCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8
    )

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("Training model...")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_int": label_to_int,
                    "int_to_label": int_to_label,
                    "num_classes": num_classes
                },
                MODEL_PATH
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("Loading best model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Evaluating best model on test set...")
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    plot_training_history(
        train_losses,
        val_losses,
        train_accs,
        val_accs
    )

    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Best Validation Loss: {best_val_loss}\n")
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_acc}\n")

    print(f"Saved best model to: {MODEL_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()