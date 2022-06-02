# YOUSSEF GEORGE

import pandas as pd
import matplotlib.pyplot as plt
import ssl
from sklearn.linear_model import LinearRegression
ssl._create_default_https_context = ssl._create_unverified_context

# Reading data from remote link
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)

# Divide the data into inputs and outputs
hours = s_data.iloc[:, :-1].values
score = s_data.iloc[:, 1].values

# Training the algorithm
regressor = LinearRegression()
regressor.fit(hours, score)

# Plotting the data and the regression line
line = regressor.coef_ * hours + regressor.intercept_
plt.scatter(hours, score)
plt.plot(hours, line)
plt.show()

# Predict score when hours of study = 9.25
predicted = regressor.predict([[9.25]])
print("No of Hours = " + str(9.25))
print("Predicted Score = " + str(predicted[0]))
