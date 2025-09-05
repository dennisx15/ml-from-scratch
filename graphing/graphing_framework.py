"""
=====================================================
 Graphing and Visualization Framework
=====================================================

This module provides functions for visualizing model performance and
image data. It includes:
 - Line plots for accuracy over training epochs
 - Confusion matrix heatmaps
 - Display of single images (grayscale)

Dependencies:
 - Plotly for interactive plots
 - Matplotlib for static image display
 - NumPy for numerical operations

Author: Dennis Alacahanli
Purpose: Graphing and visualizing model
"""

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt


# ----------------- Plot Accuracy Over Epochs ----------------- #
def plot_accuracy(accuracies):
    """
    Plot model accuracy as a function of training epochs.

    Parameters:
        accuracies (list or array): Accuracy values per epoch
    """
    fig = go.Figure()

    # Add line plot with markers
    fig.add_trace(go.Scatter(
        y=accuracies,
        mode='lines+markers',  # Show both lines and points
        name='Accuracy'
    ))

    # Update layout with titles and axis labels
    fig.update_layout(
        title="Model Accuracy Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1])  # Accuracy values between 0 and 1
    )

    # Display the interactive plot
    fig.show()


# ----------------- Plot Confusion Matrix ----------------- #
def plot_confusion_matrix(num_classes, y_true, y_pred):
    """
    Display a confusion matrix using Plotly heatmap.

    Parameters:
        num_classes (int): Number of classes
        y_true (np.ndarray): True one-hot encoded labels
        y_pred (np.ndarray): Predicted probabilities or one-hot encoded labels
    """
    # Convert one-hot encoded labels to class indices
    true_classes = np.argmax(y_true, axis=1)
    pred_classes = np.argmax(y_pred, axis=1)

    # Initialize confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Populate confusion matrix
    for t, p in zip(true_classes, pred_classes):
        conf_matrix[t, p] += 1

    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=[str(i) for i in range(num_classes)],
        y=[str(i) for i in range(num_classes)],
        colorscale='Blues',
        text=conf_matrix,
        texttemplate="%{text}",
        hoverongaps=False
    ))

    # Update layout with titles
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True"
    )

    # Display the interactive heatmap
    fig.show()


# ----------------- Display a Single Image ----------------- #
def display_image(image):
    """
    Display a single image using Matplotlib.

    Parameters:
        image (np.ndarray): 2D image array
    """
    plt.imshow(image, cmap='gray')  # Use grayscale colormap
    plt.colorbar()  # Optional: show scale
    plt.axis('off')  # Hide axis ticks and labels
    plt.show()


def display_inaccuracies(y_true, y_pred, x_test, classes):
    for i in range(len(x_test)):
        if np.argmax(y_true[i]) != np.argmax(y_pred[i]):
            img = x_test[i]
            img = img.reshape(28, 28)

            plt.title(f"True: {classes[np.argmax(y_true[i])]}, Pred: {classes[np.argmax(y_pred[i])]}", fontsize=16, color="blue", loc="left")
            display_image(img)


