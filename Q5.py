# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100

# Print accuracy in percentage
print(f"Accuracy of K-Nearest Neighbors classifier on Iris dataset: {accuracy:.2f}%")

# Convert to pandas DataFrame for further analysis and visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display DataFrame (first 5 rows)
print(df.head())

# Splitting the data based on target values
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

# Plotting Sepal Length vs Sepal Width for each class
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green", marker='+', label='Class 0')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='.', label='Class 1')
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color="red", marker='.', label='Class 2')
plt.legend()
plt.show()

# Plotting Petal Length vs Petal Width for each class
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+', label='Class 0')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='.', label='Class 1')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color="red", marker='.', label='Class 2')
plt.legend()
plt.show()

# Preparing the data for KNN model (after adding 'target' column)
X = df.drop(['target'], axis=1)
y = df['target']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Train KNN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model using the test data
accuracy = knn.score(X_test, y_test)
print(f"Accuracy of K-Nearest Neighbors classifier with 3 neighbors: {accuracy * 100:.2f}%")
