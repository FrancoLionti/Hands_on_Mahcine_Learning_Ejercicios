import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
data_root = "data-main/lifesat/lifesat.csv"
# Descargo la data del modelo, y la leo desde el path
lifesat = pd.read_csv(data_root) #leída desde el csv
X = lifesat[["GDP per capita (USD)"]].values # tomo los valores de x=gpd per capita
y = lifesat[["Life satisfaction"]].values # tomo los valores de y= satisfaccion
# Visualize the data
lifesat.plot(kind='scatter', grid=True,
x="GDP per capita (USD)", y="Life satisfaction") #metodo built in de ploteo de df
plt.axis([23_500, 62_500, 4, 9])
plt.show()
# Select a linear model
model = LinearRegression()
# Train the model
testout = model.fit(X, y)
# Make a prediction for Cyprus
X_new = [[37_655.2]] # Cyprus' GDP per capita in 2020
model.predict(X_new) # output: [[6.30165767]]
output= model.predict(X_new)
plt.plot(testout)
plt.show()