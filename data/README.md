# Dataset

Download the dataset from Kaggle:

**Link:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

**File to download:** `healthcare-dataset-stroke-data.csv`

Place it in this `data/` folder before running the notebook or script.

The dataset is not included in this repository due to Kaggle's redistribution terms.

## Dataset Details

- **Size:** 43,400 instances, 12 features
- **Target variable:** `stroke` (1 = stroke detected, 0 = no stroke)
- **Note:** Only ~4.9% of instances have stroke = 1 (imbalanced dataset — handled using SMOTE/RandomOverSampler in the code)
