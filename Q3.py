import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('Data\DV LAB\student_data.csv')

# Splitting data and training the model
X = df[['StudyHours']]
y = df['ExamScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting values for test set
y_pred = model.predict(X_test)

# Calculating Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.scatter(X_train['StudyHours'], y_train, label='Training Data', color='blue')
plt.scatter(X_test['StudyHours'], y_test, label='Test Data', color='red')
plt.plot(X_train, model.predict(X_train), color='green', label='Regression Line')
plt.xlabel('Study Hours')
plt.ylabel('Exam Scores')
plt.legend()
plt.show()

# Printing performance metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)
