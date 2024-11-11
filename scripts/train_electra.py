#!/usr/bin/env python3
"""
train.py

Training script for News Category Classification using ELECTRA.

Usage:
    python train.py --train_path data/train.jsonl --model_output model/
    python train.py --train_path data/train.jsonl --dev_path data/dev.jsonl --model_output model/
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    ModelCheckpoint
)
from transformers import (
    ElectraTokenizer,
    TFElectraForSequenceClassification,
    ElectraConfig
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

import joblib
import datetime
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
    tokenizer: ElectraTokenizer,
    label_encoder: LabelEncoder,
    max_length: int = 128
) -> Tuple[dict, np.ndarray]:
    """
    Preprocess the data by combining text fields, encoding labels, and tokenizing text.

    Args:
        df (pd.DataFrame): Input DataFrame.
        tokenizer (ElectraTokenizer): Pre-trained ELECTRA tokenizer.
        label_encoder (LabelEncoder): Encoder for the labels.
        max_length (int): Maximum length for tokenization.

    Returns:
        Tuple[dict, np.ndarray]: Tokenized inputs and encoded labels.
    """
    # Combine 'headline' and 'short_description' into a single text field, handling missing values
    df['text'] = df['headline'].fillna('') + " " + df['short_description'].fillna('')

    # Encode the categories into numerical labels
    labels = label_encoder.transform(df['category'])

    # Ensure labels are 1-D
    labels = labels.flatten().astype(np.int32)

    # Tokenize the texts using the ELECTRA tokenizer
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )

    return encodings, labels


def build_transformer_model(num_labels: int, dropout_rate: float = 0.15) -> TFElectraForSequenceClassification:
    """
    Build an ELECTRA-based classification model.

    Args:
        num_labels (int): Number of unique labels.
        dropout_rate (float): Dropout rate to be applied to the hidden layers.

    Returns:
        TFElectraForSequenceClassification: Pre-trained ELECTRA model configured for classification.
    """
    # Load the configuration with the specified dropout rates
    config = ElectraConfig.from_pretrained(
        'google/electra-base-discriminator',
        num_labels=num_labels,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate
    )

    # Load a pre-trained ELECTRA model with the custom configuration
    model = TFElectraForSequenceClassification.from_pretrained(
        'google/electra-base-discriminator',
        config=config
    )

    # Apply L2 regularization to the classifier's dense layers
    if hasattr(model.classifier, 'dense'):
        model.classifier.dense.kernel_regularizer = tf.keras.regularizers.l2(1e-7)
        model.classifier.dense.bias_regularizer = tf.keras.regularizers.l2(1e-7)
    if hasattr(model.classifier, 'out_proj'):
        model.classifier.out_proj.kernel_regularizer = tf.keras.regularizers.l2(1e-7)
        model.classifier.out_proj.bias_regularizer = tf.keras.regularizers.l2(1e-7)

    return model

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a News Category Classification model using ELECTRA.")
    parser.add_argument('--train_path', type=Path, required=True, help='Path to train.jsonl')
    parser.add_argument('--dev_path', type=Path, required=False, help='Path to dev.jsonl')
    parser.add_argument('--model_output', type=Path, required=True, help='Directory to save the trained model')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Initial learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for tokenization')
    parser.add_argument('--dropout_rate', type=float, default=0.15, help='Dropout rate for the model')
    return parser.parse_args()


def create_keras_model(transformer_model: TFElectraForSequenceClassification, max_length: int) -> tf.keras.Model:
    """
    Wrap the transformer model into a Keras Model that outputs logits directly.

    Args:
        transformer_model: Pre-trained transformer model.
        max_length (int): Maximum sequence length for input.

    Returns:
        tf.keras.Model: Keras model with inputs and logits output.
    """
    # Define input layers
    input_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

    # Get transformer outputs
    transformer_outputs = transformer_model({'input_ids': input_ids, 'attention_mask': attention_mask}, training=True)
    logits = transformer_outputs.logits

    # Define the Keras model
    keras_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)

    return keras_model


def weighted_sparse_categorical_crossentropy(class_weights: tf.Tensor):
    """
    Define a weighted sparse categorical crossentropy loss function.

    Args:
        class_weights (tf.Tensor): Tensor of class weights.

    Returns:
        function: Weighted loss function.
    """
    def loss_fn(y_true, y_pred):
        # Gather the weights for the true classes
        weights = tf.gather(class_weights, tf.cast(y_true, tf.int32))

        # Compute the standard loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

        # Apply the weights
        weighted_loss = loss * weights

        return tf.reduce_mean(weighted_loss)

    return loss_fn


def main():
    args = parse_arguments()

    # Load training data
    train_df = load_data(args.train_path)
    if args.dev_path:
        # Load validation data
        dev_df = load_data(args.dev_path)

    # Check for class imbalance
    print("Training class distribution:")
    print(train_df['category'].value_counts())
    if args.dev_path:
        print("\nValidation class distribution:")
        print(dev_df['category'].value_counts())

    # Initialize tokenizer and label encoder
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['category'])

    # Preprocess training data
    train_encodings, train_labels = preprocess_data(train_df, tokenizer, label_encoder, args.max_length)
    print(f"\nPreprocessed training samples: {len(train_labels)}")
    print(f"Training labels shape: {train_labels.shape}")
    if args.dev_path:
        # Preprocess validation data
        dev_encodings, dev_labels = preprocess_data(dev_df, tokenizer, label_encoder, args.max_length)
        print(f"Preprocessed validation samples: {len(dev_labels)}")
        print(f"Validation labels shape: {dev_labels.shape}")

    # Compute class weights to handle potential class imbalance
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    print("\nClass weights:", class_weights)

    # Convert class weights to a Tensor
    class_weights_tensor = tf.constant(list(class_weights.values()), dtype=tf.float32)

    # Build the transformer model
    transformer_model = build_transformer_model(len(label_encoder.classes_), dropout_rate=args.dropout_rate)

    # Wrap the transformer model into a Keras model
    keras_model = create_keras_model(transformer_model, args.max_length)

    # Define the optimizer with Adam for proper training
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate
    )

    # Define the weighted loss function
    loss = weighted_sparse_categorical_crossentropy(class_weights_tensor)

    # Define metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    ]

    # Compile the model
    keras_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Create datasets for training
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask']
        },
        train_labels
    )).shuffle(buffer_size=10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    if args.dev_path:
        # Create dataset for validation
        dev_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': dev_encodings['input_ids'],
                'attention_mask': dev_encodings['attention_mask']
            },
            dev_labels
        )).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            filepath=str(args.model_output / 'best_model.h5'),
            monitor='val_accuracy' if args.dev_path else 'loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        TensorBoard(log_dir=f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", histogram_freq=1),
    ]

    # Add a learning rate logger to monitor LR changes
    class LearningRateLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.lr.numpy()
            print(f"Epoch {epoch+1}: Learning rate is {lr:.6f}")

    callbacks.append(LearningRateLogger())

    # Train the model
    print("Starting training...")
    if args.dev_path:
        history = keras_model.fit(
            train_dataset,
            validation_data=dev_dataset,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = keras_model.fit(
            train_dataset,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )

    # Save the model and label encoder
    args.model_output.mkdir(parents=True, exist_ok=True)
    keras_model.save_pretrained(args.model_output)
    tokenizer.save_pretrained(args.model_output)
    joblib.dump(label_encoder, args.model_output / 'label_encoder.joblib')

    print(f"Model saved to {args.model_output}")

if __name__ == '__main__':
    main()
