#!/usr/bin/env python3
"""
eval.py

Evaluation script for News Category Classification.

Usage:
    python eval.py --model_path model/ --eval_path data/dev.jsonl
"""

import argparse
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load JSONL data into a pandas DataFrame.

    Args:
        file_path (Path): Path to the JSONL file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_json(file_path, lines=True, encoding='utf-8')


def preprocess_data(
    df: pd.DataFrame,
    tokenizer: RobertaTokenizer,
    label_encoder: LabelEncoder,
    max_length: int = 128
) -> Tuple[dict, np.ndarray]:
    """
    Preprocess the data by combining text fields, encoding labels, and tokenizing text.

    Args:
        df (pd.DataFrame): Input DataFrame.
        tokenizer (RobertaTokenizer): Pre-trained RoBERTa tokenizer.
        label_encoder (LabelEncoder): Encoder for the labels.
        max_length (int): Maximum length for tokenization.

    Returns:
        Tuple[dict, np.ndarray]: Tokenized inputs and encoded labels.
    """
    # Combine 'headline' and 'short_description' into a single text field, handling missing values
    df['text'] = df['headline'].fillna('') + " " + df['short_description'].fillna('')

    # Encode the categories into numerical labels
    labels = label_encoder.transform(df['category'])

    # Ensure labels are 1-D and
    labels = labels.flatten().astype(np.int32)

    # Tokenize the texts using the RoBERTa tokenizer
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )

    return encodings, labels

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    save_path: Path = None
):
    """
    Plot a confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        classes (List[str]): List of class names.
        save_path (Path, optional): Path to save the plot. Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate the trained model on evaluation data.")
    parser.add_argument('--model_path', type=Path, required=True, help='Path to the saved model directory)')
    parser.add_argument('--eval_path', type=Path, required=True, help='Path to evaluation JSONL file')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load the label encoder
    label_encoder = joblib.load(args.model_path / 'label_encoder.joblib')

    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(str(args.model_path))

    # Load the model
    model = TFRobertaForSequenceClassification.from_pretrained(str(args.model_path))

    # Load evaluation data
    eval_df = load_data(args.eval_path)

    # Preprocess data
    encodings, true_labels = preprocess_data(eval_df, tokenizer, label_encoder, max_length=128)

    # Create TensorFlow dataset
    eval_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        },
        true_labels
    )).batch(16)

    # Predict
    predictions = model.predict(eval_dataset)
    predicted_labels = np.argmax(predictions.logits, axis=1)

    # Generate classification report
    report = classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_)
    print("Classification Report:")
    print(report)

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, classes=label_encoder.classes_, save_path=args.model_path / 'confusion_matrix.png')


if __name__ == '__main__':
    main()
