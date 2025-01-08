import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Loading the data
df = pd.read_csv('Data\DV LAB\Solar panel.csv')

# Extracting only required columns (make sure these columns exist in the data)
df = df.iloc[:, [1, 2]]  # Make sure you extract the correct columns based on your dataset

# Splitting data into Training and Testing.
X = df[['Temperature']]
y = df['Efficiency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Plotting the train data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.xlabel('Temperature')
plt.ylabel('Efficiency')
plt.title('Training Data')
plt.show()

# Plotting the test data
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, model.predict(X_test), color='blue')
plt.xlabel('Temperature')
plt.ylabel('Efficiency')
plt.title('Test Data')
plt.show()

# F and T test (using statsmodels for statistical analysis)
X = df[['Temperature']]
Y = df['Efficiency']
X = sm.add_constant(X)  # Adds constant for the intercept in the model

# Fit the model using OLS (Ordinary Least Squares)
ols_model = sm.OLS(Y, X).fit()

# Getting the F-statistic and p-value for F-test
f_stat = ols_model.fvalue
f_p_value = ols_model.f_pvalue

# Getting the t-statistic and p-value for Temperature coefficient
t_stat = ols_model.tvalues['Temperature']
t_p_value = ols_model.pvalues['Temperature']

# Printing the results
print(f"F-statistic: {f_stat:.2f}")
print(f"F-p-value: {f_p_value:.4f}")
print(f"t-statistic for Temperature: {t_stat:.2f}")
print(f"t-p-value for Temperature: {t_p_value:.4f}")
