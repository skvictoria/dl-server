from flask import Flask, render_template, jsonify, request
from sklearn.manifold import TSNE
import torch
import numpy as np
import io
import base64
import random
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

app = Flask(__name__)

# Global data storage for training information and embeddings
training_data = {"epoch": 0, "loss": 0.0, "accuracy": 0.0}
latest_embeddings = None
latest_labels = None

# ---------- Routes ---------- #

@app.route('/')
def home():
    """
    Home page route - displays a simple message or web page.
    """
    return render_template('index.html')  # Add your UI in templates/index.html

@app.route('/data', methods=['GET'])
def get_data():
    """
    Endpoint to provide training data (epoch, loss, accuracy) in JSON format.
    """
    return jsonify(training_data)

@app.route('/update_tsne', methods=['POST'])
def update_tsne():
    """
    Endpoint to update embeddings and labels in real-time.
    Accepts JSON payload with 'embeddings' and 'labels'.
    """
    global latest_embeddings, latest_labels
    try:
        data = request.get_json()
        latest_embeddings = np.array(data['embeddings'])  # Update embeddings
        latest_labels = np.array(data['labels'])          # Update labels
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "failure", "message": str(e)}), 400

@app.route('/tsne', methods=['GET'])
def get_tsne_plot():
    """
    Endpoint to generate t-SNE visualization using Plotly.
    Returns an interactive HTML visualization.
    """
    if latest_embeddings is None or latest_labels is None:
        return "No t-SNE data available. Please update embeddings first.", 400

    tsne_html = generate_tsne_plotly(latest_embeddings, latest_labels)
    return tsne_html

# ---------- Helper Functions ---------- #

def generate_tsne_plotly(embeddings, labels):
    """
    Generates a t-SNE visualization using Plotly.
    Args:
        embeddings: numpy array of shape [N, D] - input embeddings.
        labels: numpy array of shape [N] - class labels.

    Returns:
        HTML of the t-SNE visualization.
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create DataFrame for visualization
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })

    # Generate Plotly figure
    fig = px.scatter(
        df, x='x', y='y', color=df['label'].astype(str),
        title="Real-Time t-SNE Visualization",
        labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    return fig.to_html()

# ---------- Main Execution ---------- #

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=6000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

