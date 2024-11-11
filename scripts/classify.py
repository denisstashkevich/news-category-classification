#!/usr/bin/env python3
"""
classify.py

Classification script for News Category Classification.

Usage:
    python classify.py --model_path model --input_path data/test.jsonl --output_path results/predictions.jsonl
"""

import argparse
from pathlib import Path
import json
import sys
from typing import Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib


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
    max_length: int = 128
) -> tf.data.Dataset:
    """
    Preprocess the data by combining text fields and tokenizing text.

    Args:
        df (pd.DataFrame): Input DataFrame.
        tokenizer (RobertaTokenizer): Pre-trained RoBERTa tokenizer.
        max_length (int): Maximum length for tokenization.

    Returns:
        tf.data.Dataset: TensorFlow dataset.
    """
    # Combine 'headline' and 'short_description' into a single text field, handling missing values
    df['text'] = df['headline'].fillna('') + " " + df['short_description'].fillna('')

    # Tokenize the texts using the RoBERTa tokenizer
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )

    # Convert lists to numpy arrays for TensorFlow
    input_ids = np.array(encodings['input_ids'])
    attention_mask = np.array(encodings['attention_mask'])

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    )).batch(16).prefetch(tf.data.AUTOTUNE)
    return dataset


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Classify news articles using a trained RoBERTa model.")
    parser.add_argument('--model_path', type=Path, required=True, help='Path to the saved model directory')
    parser.add_argument('--input_path', type=Path, required=True, help='Path to input JSONL file without categories')
    parser.add_argument('--output_path', type=Path, required=True, help='Path to save predictions')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load the label encoder
    label_encoder_path = args.model_path / 'label_encoder.joblib'
    if not label_encoder_path.exists():
        print(f"Label encoder not found at {label_encoder_path}. Please ensure it exists.", file=sys.stderr)
        sys.exit(1)
    label_encoder = joblib.load(label_encoder_path)

    # Load the tokenizer
    try:
        tokenizer = RobertaTokenizer.from_pretrained(str(args.model_path))
    except Exception as e:
        print(f"Error loading the tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # Load the model
    try:
        model = TFRobertaForSequenceClassification.from_pretrained(str(args.model_path))
    except Exception as e:
        print(f"Error loading the model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load input data
    input_df = load_data(args.input_path)

    # Preprocess data
    input_dataset = preprocess_data(input_df, tokenizer, max_length=128)

    # Predict
    print("Starting prediction...")
    predictions = model.predict(input_dataset)
    predicted_labels = np.argmax(predictions.logits, axis=1)
    predicted_categories = label_encoder.inverse_transform(predicted_labels)

    # Add predictions to the data
    input_df['predicted_category'] = predicted_categories

    # Save predictions
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output_path.open('w', encoding='utf-8') as f:
            for record in input_df.to_dict(orient='records'):
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except IOError as e:
        print(f"Error writing predictions to {args.output_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Predictions saved to {args.output_path}")


if __name__ == '__main__':
    main()
