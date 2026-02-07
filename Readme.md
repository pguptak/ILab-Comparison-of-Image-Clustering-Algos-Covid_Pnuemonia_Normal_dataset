
# COVID Chest X-Ray Clustering Project

## Overview

This repository contains an unsupervised clustering study performed on a **COVID–Pneumonia–Normal chest X-ray dataset** using deep learning feature extractors and classical clustering algorithms. Deep learning models are used **only for feature extraction**, and multiple clustering algorithms are applied on the extracted features to analyze **label drift** and **cluster behavior**.

---

## Functioning Model Description

The workflow followed in this project is:

1. Chest X-ray images are passed through pretrained deep learning models
2. High-level feature vectors are extracted and stored as `.npy` files
3. Multiple clustering algorithms are applied on the extracted feature vectors
4. Cluster labels are compared with original ground truth labels
5. Percentage of label change is computed for each model–clustering combination

---

## Dataset Description

The dataset consists of chest X-ray images belonging to three classes:

* COVID
* Pneumonia
* Normal

Each image is associated with a ground truth label stored in a CSV file.

---

## Dataset Link

Dataset source:
[https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images](https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images)

---

## Repository Structure

```
.
├── clustering_labels/
│   ├── EfficientNetB0_KMeans.csv
│   ├── EfficientNetB0_DBSCAN.csv
│   ├── ResNet152_KMeans.csv
│   ├── VGG19_Spectral.csv
│   └── ... (all model + clustering combinations)
│
├── clustering_covid.ipynb
├── covid_pneumonia_normal.ipynb
├── original_labels.csv
└── README.md
```

---

## Files Description

### covid_pneumonia_normal.ipynb

Used for dataset inspection, preprocessing, and feature extraction using deep learning models.

### clustering_covid.ipynb

Applies multiple clustering algorithms on extracted features, computes label change percentages, and saves clustering outputs.

### original_labels.csv

Contains image names and original class labels.

### clustering_labels/

Contains CSV files mapping image names to cluster labels for each model–clustering combination.

---

## Requirements

Python 3.9 or above

Required libraries:

```
numpy
pandas
scikit-learn
scikit-learn-extra
hdbscan
scikit-fuzzy
tensorflow
keras
```

---

## How to Run

1. Clone the repository
2. Place the dataset images and `original_labels.csv` in the appropriate directory
3. Run feature extraction:

   ```
   covid_pneumonia_normal.ipynb
   ```
4. Run clustering:

   ```
   clustering_covid.ipynb
   ```
5. Check results in:

   * `clustering_labels/`
   * `clustering_results.csv`
