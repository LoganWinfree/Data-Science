import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import data from csv
data = pd.read_csv("car_prices.csv")

# Split the data into predictor and response data objects
X_data = data.iloc[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] # All columns identified as predictors
y_data = data.iloc[:, 14] # Sales price column

# Split the data into training/testing
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1234)

# Combine training data for EDA
train_data = pd.concat([X_train, y_train], axis=1)

# Interval - Years
plt.figure("Distribution of Vehicle Years")
plt.hist(train_data['year'], bins=np.unique(train_data['year']))
plt.title("Distribution of Vehicle Years")
plt.xlabel("Year")
#plt.xticks(np.arange(np.unique(train_data['year']).min(), np.unique(train_data['year']).max()+1, 1), rotation=-45)

# Numerical
fig, (ax1, ax2) = plt.subplots(1,2)
fig.canvas.manager.set_window_title("Distributions of Numerical Variables")
ax1.boxplot(train_data['condition'].dropna())
ax1.set_ylim(0,100)
ax1.set_xlabel("Condition")
ax2.boxplot(train_data['odometer'].dropna(), label="Odometer")
ax2.set_xlabel("Odometer")



plt.show()
