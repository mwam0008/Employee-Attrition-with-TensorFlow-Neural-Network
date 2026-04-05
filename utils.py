"""
utils.py - Visualization and helper functions for Employee Attrition App
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Required for Streamlit (no display)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)


def plot_attrition_distribution(df: pd.DataFrame):
    """Bar chart showing how many employees stayed vs left."""
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df['Attrition'].value_counts()
        colors = ['#2196F3', '#F44336']
        ax.bar(counts.index.astype(str), counts.values, color=colors)
        ax.set_title('Employee Attrition Distribution')
        ax.set_xlabel('Attrition (0 = Stayed, 1 = Left)')
        ax.set_ylabel('Count')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting attrition distribution: {e}")
        raise


def plot_training_curves(history):
    """Plot loss and accuracy curves from training history."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        axes[0].plot(history.history['loss'], color='#F44336', label='Loss')
        axes[0].set_title('Training Loss Over Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Accuracy curve
        axes[1].plot(history.history['accuracy'], color='#2196F3', label='Accuracy')
        axes[1].set_title('Training Accuracy Over Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting training curves: {e}")
        raise


def plot_confusion_matrix(cm):
    """Heatmap of the confusion matrix."""
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Stayed (0)', 'Left (1)'],
                    yticklabels=['Stayed (0)', 'Left (1)'],
                    ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting confusion matrix: {e}")
        raise


def plot_learning_rate_search(history):
    """Plot learning rate vs loss to find the sweet spot."""
    try:
        lrs = 1e-5 * (10 ** (np.arange(100) / 20))
        losses = history.history['loss']

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogx(lrs, losses, color='#9C27B0')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate vs Loss — Find the Sweet Spot')
        ax.axvline(x=0.001, color='red', linestyle='--', label='~Ideal LR (0.001)')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting learning rate search: {e}")
        raise


def plot_feature_correlation(df: pd.DataFrame):
    """Heatmap of feature correlations."""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax, linewidths=0.5)
        ax.set_title('Feature Correlation Heatmap')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error plotting correlation: {e}")
        raise
