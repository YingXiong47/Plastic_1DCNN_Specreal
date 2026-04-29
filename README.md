# Plastic Classification with 4-Channel 1D CNN (PyTorch)

## Overview

This project implements a **1D Convolutional Neural Network (1D CNN)** in **PyTorch** to classify plastic types using **NIR/SWIR spectral data**.

Each sample is treated as a **1D signal (spectrum)** and expanded into a **4-channel input**:

- Reflectance  
- Absorption  
- Absorption (1st derivative)  
- Absorption (2nd derivative)  

This allows the model to learn both raw spectral behavior and higher-order feature patterns.

---

## Key Idea

Instead of using a single spectral representation, this project stacks multiple transformations of the same signal:
Input shape (PyTorch):
(samples, channels, sequence_length)

Example:
(____ samples, 4, ____ wavelengths)


Channels:
0 → Reflectance
1 → Absorption
2 → First derivative
3 → Second derivative


---

## Dataset

This project uses:

**Holt, Z.K.; Khan, S.D.; Rodrigues, D.F.**  
*Spectral Library of Plastics mixed with Environmental Substrates*  
DOI: https://doi.org/10.5281/zenodo.14233290  

### Important

- I did **not** create this dataset  
- The dataset is used for **machine learning experimentation only**  
- Licensed under **CC BY 4.0**  

---

## My Contribution

- Built preprocessing pipeline for spectral data  
- Converted CSV data into **4-channel tensor format**  
- Implemented PyTorch **1D CNN model**  
- Trained and evaluated classification performance  

---

## Project Structure
plastic-1dcnn-spectral/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── raw/
│ ├── processed/
│
├── notebooks/
│
├── src/
│ ├── preprocess.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── utils.py
│
├── models/
│
└── results/
├── plots/
└── metrics.txt


---

## Installation

```bash
pip install -r requirements.txt

---


## Usage
1. Preprocess data
python src/preprocess.py

2. Train model
python src/train.py

3. Evaluate model
python src/evaluate.py

---


## Model Architecture
Conv1D layers → extract spectral features
Batch normalization
Pooling layers
Fully connected layers for classification

---


## Loss:
CrossEntropyLoss

---


## Limitations
Small dataset → risk of overfitting
High dimensional input (~1500 wavelengths)
Possible class imbalance

---


## Future Work
Feature selection / wavelength reduction
Cross-validation
Compare with classical ML models
Try transformer-based architectures

---


## License
Original dataset: CC BY 4.0
This repository: research / educational use

---


## Contact
Name: Ismail Muhammad 
Email: izzymuhammad3@outlook.com
