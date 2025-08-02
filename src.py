import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

#Importing the data set
dataset = pd.read_csv('data.csv')
print('The data set is: ',dataset)


# Creating the gradient descent function
def gradient_descent(m_now,b_now,points,L):
    for i in range(len(points)):
        x = points.iloc[i].X
        y = points.iloc[i].Y

        n = len(points)

        m_gradient = -(2/n) * x *(y - (m_now * x + b_now))
        b_gradient = -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.0001
epochs = 1000

# Calculating the values of m,b
for i in range(epochs):
    m,b = gradient_descent(m,b,dataset,L)

y_pred = np.array((m * dataset.X) + b)
#Visualising the original data set and also the regression line 
plt.figure(figsize=(15,10))
plt.scatter(dataset.X,dataset.Y)
plt.xlabel('Study Time')
plt.ylabel('Exam score')
plt.plot(dataset.X,y_pred)
plt.show()

#Calculating the accuracy of the regression
print(f'The R-squared score is: {r2_score(dataset.Y,y_pred)}')