#YOUSSEF GEORGE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reading data from local csv file
iris_data = pd.read_csv('Iris.csv')

# Using the elbow method to predict the optimum number of clusters
# Get the value of "within cluster sum of squares" (wcss) for range of numbers 1-10
# Plot wcss against the different number of clusters 1-10 to apply the elbow method
x = iris_data.iloc[:, [0, 1, 2, 3]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel('# of clusters')
plt.ylabel('Within cluster sum of squares')
plt.show()


# Optimal number of clusters is 3
# Use 3 to predict the labels and scatter them to represent them visually as clusters
kmeans = KMeans(n_clusters = 3)
predicted_labels = kmeans.fit_predict(x)
plt.scatter(x[predicted_labels == 0, 0], x[predicted_labels == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[predicted_labels == 1, 0], x[predicted_labels == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[predicted_labels == 2, 0], x[predicted_labels == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

plt.legend()
plt.show()
