"""
model.py - All ML model logic for Employee Attrition Classification
Uses TensorFlow / Keras neural network for binary classification
"""

import logging
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(filepath: str):
    """Load CSV and separate features from target (Attrition)."""
    try:
        logging.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded. Shape: {df.shape}")

        Y = df['Attrition']
        X = df.drop(columns=['Attrition'])

        logging.info(f"Features: {X.shape[1]}, Target classes: {Y.unique()}")
        return df, X, Y
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def scale_and_split(X, Y, test_size=0.2, random_state=1):
    """Scale features with StandardScaler and split into train/test sets."""
    try:
        logging.info("Scaling and splitting data...")

        # Scale first, then split
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        x_train, x_test, y_train, y_test = train_test_split(
            X_scaled, Y,
            test_size=test_size,
            random_state=random_state,
            stratify=Y
        )

        logging.info(f"Train size: {x_train.shape}, Test size: {x_test.shape}")
        return x_train, x_test, y_train, y_test, sc
    except Exception as e:
        logging.error(f"Scaling/splitting failed: {e}")
        raise


def build_model(neurons=3, extra_layer=False, learning_rate=0.01, activation='sigmoid'):
    """
    Build a TensorFlow Sequential neural network for binary classification.

    Args:
        neurons: Number of neurons in the hidden layer
        extra_layer: Whether to add an additional hidden layer
        learning_rate: Learning rate for SGD optimizer
        activation: Activation function for output layer
    """
    try:
        logging.info(f"Building model: neurons={neurons}, extra_layer={extra_layer}, lr={learning_rate}")
        tf.keras.utils.set_random_seed(42)

        layers = []
        layers.append(tf.keras.layers.Dense(neurons))

        if extra_layer:
            layers.append(tf.keras.layers.Dense(neurons))

        # Output layer - sigmoid for binary classification
        layers.append(tf.keras.layers.Dense(1, activation=activation))

        model = tf.keras.Sequential(layers)

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=['accuracy']
        )

        logging.info("Model built and compiled successfully.")
        return model
    except Exception as e:
        logging.error(f"Model build failed: {e}")
        raise


def train_model(model, x_train, y_train, epochs=100, batch_size=32):
    """Train the model and return the training history."""
    try:
        logging.info(f"Training model for {epochs} epochs...")
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        logging.info("Training complete.")
        return history
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def evaluate_model(model, x_test, y_test):
    """Evaluate model on test set and return metrics."""
    try:
        logging.info("Evaluating model on test set...")

        # Get raw probability predictions
        y_probs = model.predict(x_test, verbose=0)

        # Round to binary 0 or 1
        y_preds = tf.round(y_probs).numpy().flatten().astype(int)

        acc = accuracy_score(y_test, y_preds)
        report = classification_report(y_test, y_preds, output_dict=True)
        cm = confusion_matrix(y_test, y_preds)

        logging.info(f"Test Accuracy: {acc:.4f}")
        return acc, report, cm, y_preds, y_probs
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


def find_best_learning_rate(x_train, y_train, neurons=3):
    """
    Use a learning rate scheduler callback to find the best learning rate.
    Returns training history across a range of learning rates.
    """
    try:
        logging.info("Finding best learning rate...")
        tf.keras.utils.set_random_seed(42)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(neurons),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.SGD(),
            metrics=["accuracy"]
        )

        # Learning rate scheduler callback
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-5 * (10 ** (epoch / 20))
        )

        history = model.fit(
            x_train, y_train,
            epochs=100,
            verbose=0,
            callbacks=[lr_scheduler]
        )

        logging.info("Learning rate search complete.")
        return history
    except Exception as e:
        logging.error(f"Learning rate search failed: {e}")
        raise
