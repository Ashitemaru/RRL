import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import pandas as pd
# from load import load_csv_to_pandas_csv as load
# from preprocess import data_cleaning

filename = './data/HousingData.csv'
data = pd.read_csv(filename, header=0)
data = data.dropna()
# data = load(filename)
# data = data_cleaning(data)
data = np.array(data)

features = data[:, :-1]
labels = data[:, -1]
# print(features.shape)
# print(labels.shape)
pca = decomposition.PCA(n_components=2)
pcs = pca.fit_transform(features)

label_0_pcs = pcs[labels==False, :]
label_1_pcs = pcs[labels==True, :]

print(label_0_pcs.shape)
print(label_1_pcs.shape)

plt.scatter(label_0_pcs[:, 0], label_0_pcs[:, 1], s=2, c="blue")
plt.scatter(label_1_pcs[:, 0], label_1_pcs[:, 1], s=2, c="red")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()