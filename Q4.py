# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Loading the dataset
df = pd.read_csv('Data\DV LAB\sales (part A).csv')
df.head()  # Displaying first few rows

# Training a linear regression model on multiple features
reg = LinearRegression()
reg.fit(df[['AdvertisingExpenditure', 'StoreLocation', 'Competition']], df['SalesRevenue'])

# Model coefficients and intercept
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# Analyzing the relationship between AdvertisingExpenditure and SalesRevenue
x1 = df[['AdvertisingExpenditure']]
y1 = df[['SalesRevenue']]

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=42)

# Training and visualizing the model for AdvertisingExpenditure
model = LinearRegression()
model.fit(x_train, y_train)

# Printing the coefficients and intercept after training
print(f"Coefficients for AdvertisingExpenditure model: {model.coef_}")
print(f"Intercept for AdvertisingExpenditure model: {model.intercept_}")

# Plotting the data
plt.scatter(x_train, y_train, color='red')  # Training data
plt.scatter(x_test, y_test, color='blue')  # Testing data
plt.plot(x_train, model.predict(x_train), color='green')  # Regression line
plt.xlabel('AdvertisingExpenditure')
plt.ylabel('SalesRevenue')
plt.show()

# Repeating analysis for StoreLocation and Competition

# StoreLocation
x2 = df[['StoreLocation']]
x_train, x_test, y_train, y_test = train_test_split(x2, y1, test_size=0.3, random_state=42)
model.fit(x_train, y_train)

# Printing the coefficients and intercept for StoreLocation
print(f"Coefficients for StoreLocation model: {model.coef_}")
print(f"Intercept for StoreLocation model: {model.intercept_}")

# Plotting for StoreLocation
plt.scatter(x_train, y_train, color='red')
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_train, model.predict(x_train), color='green')
plt.xlabel('StoreLocation')
plt.ylabel('SalesRevenue')
plt.show()

# Competition
x3 = df[['Competition']]
x_train, x_test, y_train, y_test = train_test_split(x3, y1, test_size=0.3, random_state=42)
model.fit(x_train, y_train)

# Printing the coefficients and intercept for Competition
print(f"Coefficients for Competition model: {model.coef_}")
print(f"Intercept for Competition model: {model.intercept_}")

# Plotting for Competition
plt.scatter(x_train, y_train, color='red')
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_train, model.predict(x_train), color='green')
plt.xlabel('Competition')
plt.ylabel('SalesRevenue')
plt.show()

# F test using statsmodels
X = df[['AdvertisingExpenditure', 'Competition', 'StoreLocation']]
Y = df['SalesRevenue']
X = sm.add_constant(X)

# Fit the model using OLS (Ordinary Least Squares)
ols_model = sm.OLS(Y, X).fit()

# Extracting statistics
f_stat = ols_model.fvalue
t_stat_advertising = ols_model.tvalues['AdvertisingExpenditure']
t_stat_competition = ols_model.tvalues['Competition']
t_stat_location = ols_model.tvalues['StoreLocation']

# Printing the results
print(f"F-statistic: {f_stat:.2f}")
print(f"t-statistic for AdvertisingExpenditure: {t_stat_advertising:.2f}")
print(f"t-statistic for Competition: {t_stat_competition:.2f}")
print(f"t-statistic for StoreLocation: {t_stat_location:.2f}")
