import matplotlib.pyplot as plt
import pandas as pd

# Reading a file already created using Excel
df = pd.read_csv('Data\DV LAB\Plants.csv')
print(df.head())  # Just to see the first few rows of the CSV (optional)

# Or you can create a sample dataset directly
# data = {
#     'plant_name': ['Fern', 'Cactus', 'Bamboo', 'Rose', 'Tulip', 'Daisy', 'Sunflower', 'Lily', 'Orchid', 'Maple'],
#     'sunlight_exposure': [5, 10, 12, 8, 6, 7, 14, 11, 9, 5],  # hours of sunlight
#     'plant_height': [30, 150, 200, 60, 50, 40, 180, 70, 40, 20]  # height in cm
# }
# df1 = pd.DataFrame(data)

# Reducing the dataframe to two columns
df = df[['sunlight_exposure', 'plant_height']]  # Assuming you want to focus only on these columns

# Visualize the relationship between sunlight exposure and plant height using a scatterplot
plt.scatter(df['sunlight_exposure'], df['plant_height'], color="black", marker="*")
plt.title('Relationship between sunlight exposure and plant height')
plt.xlabel('Sunlight Exposure (hours)')
plt.ylabel('Plant Height (cm)')
plt.grid(True)
plt.show()

# III. Calculate the correlation coefficient between sunlight exposure and plant height.
# Is the correlation positive or negative? Is it strong or weak?
correlation = df['sunlight_exposure'].corr(df['plant_height'])
print(f"Correlation between sunlight exposure and plant height: {correlation}")

# d. Based on the correlation coefficient, can we conclude that there is a significant association between sunlight exposure and plant growth rate?
threshold = 0.7
if abs(correlation) >= threshold:
    print("There is a significant association between sunlight exposure and plant growth rate.")
else:
    print("There is no significant association between sunlight exposure and plant growth rate.")
