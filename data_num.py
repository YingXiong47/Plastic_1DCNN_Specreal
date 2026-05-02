import json
import os 

PATH = os.path.join("data/processed/processed_4channel_spectral_dataset.json")
with open (PATH, "r") as f:
    data = json.load(f)

print("Train samples:", len(data["train"]["X"]))
print("Val samples:", len(data["val"]["X"]))
print("Test samples:", len(data["test"]["X"]))