from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

n_dots = 100
n_train_datasets = 1000000

model = Pipeline([
    ('scaler', StandardScaler()),  # Нормализуем данные
    ('mlp', MLPClassifier(
        hidden_layer_sizes = (n_dots, n_dots//2),  # Архитектура сети
        activation='relu',
        max_iter=1000,
        verbose=True
    ))
])

X_train = []
Y_train = []

for i in range(n_train_datasets):

    n_centers = randint(1,5)
    raw_data = make_blobs(n_samples = n_dots, n_features = 2, centers = n_centers, cluster_std = 1)

    X_train.append(raw_data[0].flatten().tolist())
    Y_train.append(n_centers)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# model = MLPClassifier(hidden_layer_sizes=(n_dots, n_dots//2))
model.fit(X_train, Y_train)
print("Accuracy:", model.score(X_train, Y_train))

raw_data = make_blobs(n_samples = n_dots, n_features = 2, centers = 5, cluster_std = 1)
model_clusters = KMeans(n_clusters = model.predict([raw_data[0].flatten().tolist()])[0])
model_clusters.fit(raw_data[0])

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('Predictions')
ax1.scatter(raw_data[0][:,0], raw_data[0][:,1],c=model_clusters.labels_)
ax2.set_title('Real data')
ax2.scatter(raw_data[0][:,0], raw_data[0][:,1],c=raw_data[1]);
plt.show()