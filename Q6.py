from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv("Data\DV LAB\income_clustering.csv")
print(df.head())

# Scatter plot of Age vs Income
plt.scatter(df.Age, df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

# Perform KMeans clustering with 3 clusters
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])

# Add the predicted cluster values to the dataframe
df['cluster'] = y_predicted
print(df.head())

# Create separate dataframes for each cluster
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Scatter plot of clusters
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')

# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroids')

# Display the plot with labels
plt.legend()
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

# Elbow method to determine optimal k (number of clusters)
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

# Plotting the elbow curve
plt.xlabel('K')
plt.ylabel('Sum of Squared Error (SSE)')
plt.plot(k_rng, sse)
plt.show()
