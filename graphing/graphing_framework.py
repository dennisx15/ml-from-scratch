import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy(accuracies):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=accuracies,
        mode='lines+markers',  # lines and points
        name='Accuracy'
    ))

    # Layout
    fig.update_layout(
        title="Model Accuracy Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1])  # Accuracy between 0 and 1
    )

    fig.show()

def plot_confusion_matrix(num_classes, y_true, y_pred):
    true_classes = np.argmax(y_true, axis=1)
    pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(true_classes, pred_classes):
        conf_matrix[t, p] += 1

    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=[str(i) for i in range(num_classes)],
        y=[str(i) for i in range(num_classes)],
        colorscale='Blues',
        text=conf_matrix,
        texttemplate="%{text}",
        hoverongaps=False
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True"
    )

    fig.show()

def display_image(image):
    plt.imshow(image, cmap='gray')  # cmap='gray' for grayscale
    plt.colorbar()  # optional, shows scale
    plt.axis('off')  # hide axes
    plt.show()

