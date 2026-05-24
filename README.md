# PV Fault Detection & Feature Selection Comparison

This project implements a machine learning pipeline to detect faults in Photovoltaic (PV) systems, specifically classifying conditions into **Partial Shading (PS)**, **String Mismatch (MM)**, and **Normal** conditions. 

The core of this project is a comparison of three distinct feature selection algorithms used to optimize a Random Forest Classifier.

## Project Pipeline

1. **Data Loading & Cleaning (`data_loader.py`)**
   - Loads raw PV dataset from Excel.
   - Cleans numeric columns and normalizes fault labels.

2. **Feature Engineering (`feature_engineering.py`)**
   - Derives synthetic physical features (e.g., voltage/current ratios) to expand the feature space.
   - Automatically drops near-constant features.

3. **Feature Selection (`feature_selection.py`)**
   - Implements three optimization algorithms to select the most discriminative features:
     - **CRSA** (Chaotic Reptile Search Algorithm)
     - **PSO** (Particle Swarm Optimization)
     - **SOM-GA** (Self-Organizing Maps + Genetic Algorithm)

4. **Classification & Evaluation (`classifier.py` & `main.py`)**
   - Uses a **Random Forest Classifier** to evaluate the selected features from each algorithm.
   - Outputs a comparative report including Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## Setup Instructions

Run this command to install the necessary libraries:
```bash
pip install -r requirement.txt
```

To verify the installation, run:
```bash
pip list
```

## Running the Pipeline

Execute the main orchestrator script:
```bash
python main.py
```
*(Note for Windows users: If you experience emoji rendering errors in the console, run `$env:PYTHONIOENCODING="utf-8"; python main.py`)*
