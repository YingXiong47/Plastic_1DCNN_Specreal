import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

REFLECTANCE_PATH = os.path.join(RAW_DIR, "__Mean_Spectra_Reflectance.csv")
ABSORPTION_PATH = os.path.join(RAW_DIR, "__Mean_Spectra_Absorption.csv")
DER1_PATH = os.path.join(RAW_DIR, "__Mean_Spectra_Absorption_der1.csv")
DER2_PATH = os.path.join(RAW_DIR, "__Mean_Spectra_Absorption_der2.csv")

JSON_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "processed_4channel_spectral_dataset.json")
NPZ_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "processed_4channel_spectral_dataset.npz")


def load_csvs():
    df_ref = pd.read_csv(REFLECTANCE_PATH)
    df_abs = pd.read_csv(ABSORPTION_PATH)
    df_der1 = pd.read_csv(DER1_PATH)
    df_der2 = pd.read_csv(DER2_PATH)
    return df_ref, df_abs, df_der1, df_der2


def check_columns_match(df_ref, df_abs, df_der1, df_der2):
    assert list(df_ref.columns) == list(df_abs.columns), "Reflectance and absorption columns do not match."
    assert list(df_ref.columns) == list(df_der1.columns), "Reflectance and derivative 1 columns do not match."
    assert list(df_ref.columns) == list(df_der2.columns), "Reflectance and derivative 2 columns do not match."


def build_dataset(df_ref, df_abs, df_der1, df_der2):
    wavelengths = df_ref.iloc[:, 0].to_numpy(dtype=np.float32)
    sample_columns = df_ref.columns[1:]

    X = []
    y = []
    sample_names = []

    for col in sample_columns:
        parts = col.split("_")

        if len(parts) < 3 or parts[1].strip() == "":
            print(f"Skipping bad column: {col}")
            continue

        class_label = parts[1].strip()

        ref_vector = df_ref[col].to_numpy(dtype=np.float32)
        abs_vector = df_abs[col].to_numpy(dtype=np.float32)
        der1_vector = df_der1[col].to_numpy(dtype=np.float32)
        der2_vector = df_der2[col].to_numpy(dtype=np.float32)

        sample_4_channel = np.stack(
            [ref_vector, abs_vector, der1_vector, der2_vector],
            axis=-1
        )

        X.append(sample_4_channel)
        y.append(class_label)
        sample_names.append(col)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y, wavelengths, sample_names


def encode_labels(y):
    unique_classes = sorted(np.unique(y))

    label_to_int = {label: i for i, label in enumerate(unique_classes)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    y_encoded = np.array([label_to_int[label] for label in y], dtype=np.int64)

    return y_encoded, label_to_int, int_to_label


def split_dataset(X, y_encoded, sample_names):
    X_train, X_temp, y_train, y_temp, names_train, names_temp = train_test_split(
        X,
        y_encoded,
        sample_names,
        test_size=0.30,
        random_state=42,
        stratify=y_encoded
    )

    X_val, X_test, y_val, y_test, names_val, names_test = train_test_split(
        X_temp,
        y_temp,
        names_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, names_train, names_val, names_test


def normalize_data(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


def convert_for_pytorch(X):
    """
    PyTorch Conv1D expects:
    (samples, channels, sequence_length)

    Current shape:
    (samples, sequence_length, channels)

    So we transpose.
    """
    return np.transpose(X, (0, 2, 1))


def save_json(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    names_train,
    names_val,
    names_test,
    wavelengths,
    label_to_int,
    int_to_label,
    mean,
    std
):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    processed_data = {
        "description": "4-channel spectral dataset for PyTorch 1D CNN plastic classification.",
        "channels": [
            "reflectance",
            "absorption",
            "absorption_derivative_1",
            "absorption_derivative_2"
        ],
        "input_shape_pytorch": list(X_train.shape[1:]),
        "input_format": "(samples, channels, sequence_length)",
        "label_to_int": label_to_int,
        "int_to_label": {str(k): v for k, v in int_to_label.items()},
        "wavelengths": wavelengths.tolist(),
        "normalization": {
            "mean": mean.flatten().tolist(),
            "std": std.flatten().tolist()
        },
        "train": {
            "sample_names": names_train,
            "X": X_train.tolist(),
            "y": y_train.tolist()
        },
        "val": {
            "sample_names": names_val,
            "X": X_val.tolist(),
            "y": y_val.tolist()
        },
        "test": {
            "sample_names": names_test,
            "X": X_test.tolist(),
            "y": y_test.tolist()
        }
    }

    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(processed_data, f, indent=2)

    print(f"Saved JSON dataset to: {JSON_OUTPUT_PATH}")


def save_npz(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    wavelengths
):
    np.savez_compressed(
        NPZ_OUTPUT_PATH,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        wavelengths=wavelengths
    )

    print(f"Saved NPZ dataset to: {NPZ_OUTPUT_PATH}")


def main():
    print("Loading CSV files...")
    df_ref, df_abs, df_der1, df_der2 = load_csvs()

    print("Checking column alignment...")
    check_columns_match(df_ref, df_abs, df_der1, df_der2)

    print("Building 4-channel dataset...")
    X, y, wavelengths, sample_names = build_dataset(df_ref, df_abs, df_der1, df_der2)

    print("Original X shape:", X.shape)
    print("Original format: (samples, sequence_length, channels)")
    print("Classes:", np.unique(y))

    print("Encoding labels...")
    y_encoded, label_to_int, int_to_label = encode_labels(y)
    print("Label mapping:", label_to_int)

    print("Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, names_train, names_val, names_test = split_dataset(
        X,
        y_encoded,
        sample_names
    )

    print("Normalizing data...")
    X_train, X_val, X_test, mean, std = normalize_data(X_train, X_val, X_test)

    print("Converting to PyTorch Conv1D format...")
    X_train = convert_for_pytorch(X_train)
    X_val = convert_for_pytorch(X_val)
    X_test = convert_for_pytorch(X_test)

    print("Final X_train shape:", X_train.shape)
    print("Final X_val shape:", X_val.shape)
    print("Final X_test shape:", X_test.shape)
    print("Expected format: (samples, channels, sequence_length)")

    print("Saving processed dataset...")
    save_json(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        names_train,
        names_val,
        names_test,
        wavelengths,
        label_to_int,
        int_to_label,
        mean,
        std
    )

    save_npz(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        wavelengths
    )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()