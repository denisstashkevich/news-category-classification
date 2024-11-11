# News Category Classification

## Overview

This project builds a text classification model to categorize news articles into predefined categories using the RoBERTa (and optionally ELECTRA) transformer model. It includes scripts for training, evaluating, and classifying news articles.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Classification](#classification)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Installation

Clone the repository:

```bash
git clone https://github.com/denisstashkevich/news-category-classification.git
cd news-category-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the RoBERTa model with optional hyperparameter tuning using Optuna.

**Without Tuning:**

```bash
python train.py --train_path data/train.jsonl --dev_path data/dev.jsonl --model_output model/
```

**With Hyperparameter Tuning:**

```bash
python train.py --train_path data/train.jsonl --dev_path data/dev.jsonl --model_output model/ --tune
```

**Arguments:**

- `--train_path`: Path to training data (`train.jsonl`).
- `--dev_path`: Path to validation data (`dev.jsonl`).
- `--model_output`: Directory to save the trained model.
- `--batch_size`: (Optional) Training batch size. Default: `8`.
- `--epochs`: (Optional) Number of training epochs. Default: `12`.
- `--learning_rate`: (Optional) Initial learning rate. Default: `3e-5`.
- `--weight_decay`: (Optional) Weight decay factor. Default: `1e-6`.
- `--max_length`: (Optional) Max token length. Default: `128`.
- `--dropout_rate`: (Optional) Dropout rate. Default: `0.3`.
- `--tune`: (Optional) Enable hyperparameter tuning.

### Evaluation

Evaluate the trained model on validation data.

```bash
python eval.py --model_path model/ --eval_path data/dev.jsonl
```

**Arguments:**

- `--model_path`: Path to the saved model directory.
- `--eval_path`: Path to evaluation data (`dev.jsonl`).

**Outputs:**

- **Classification Report:** Precision, recall, F1-score for each category.
- **Confusion Matrix:** Saved as `confusion_matrix.png` in the model directory.

### Classification

Classify new news articles and save predictions.

```bash
python classify.py --model_path model/ --input_path data/test.jsonl --output_path results/predictions.jsonl
```

**Arguments:**

- `--model_path`: Path to the saved model directory.
- `--input_path`: Path to input JSONL file without categories (`test.jsonl`).
- `--output_path`: Path to save predictions (`predictions.jsonl`).

**Outputs:**

- **Predictions File:** JSONL file with `predicted_category` for each article.

## Project Structure

```javascript
news-category-classification/
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── model/
│   ├── tf_model.h5
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── special_tokens_map.json
│   ├── confusion_matrix.png
│   └── label_encoder.joblib
├── results/
│   └── predictions.jsonl
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── classify.py
├── requirements.txt
└── README.md
```

- **data**: Training, validation, and test datasets in JSONL format.
- **model**: Trained model, tokenizer, and label encoder.
- **results**: Classification predictions.
- **scripts**: Python scripts for training, evaluation, and classification.
- **requirements.txt**: Project dependencies.
- **README.md**: Project documentation.

## Dependencies

All dependencies are listed in `requirements.txt`. Key packages:

- **pandas~=2.2.3**: Data manipulation.
- **numpy~=1.23.5**: Numerical operations.
- **tensorflow~=2.11.0**: Machine learning framework.
- **joblib~=1.4.2**: Object serialization.
- **matplotlib~=3.9.2**: Plotting.
- **seaborn~=0.13.2**: Data visualization.
- **transformers~=4.30.0**: NLP models.
- **scikit-learn~=1.5.2**: Machine learning utilities.
- **optuna~=4.0.0**: Hyperparameter optimization.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Follow these steps:

Fork the repository.

Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

Commit your changes:

```bash
git commit -m "Add your descriptive commit message"
```

Push to the branch:

```bash
git push origin feature/your-feature-name
```

Create a pull request, providing details about your changes.

