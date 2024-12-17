import requests
import numpy as np

embeddings = np.random.rand(100, 128).tolist()
labels = np.random.randint(0, 10, 100).tolist()

response = requests.post("http://localhost:5000/update_tsne", json={
    "embeddings": embeddings,
    "labels": labels
})
print(response.json())