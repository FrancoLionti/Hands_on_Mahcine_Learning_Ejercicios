import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Descargo la data del modelo, y la leo desde el path
data_root = "/home/franco/Code projects/Hands on Mahcine Learning - ejercicios/E1/data-main/lifesat/lifesat.csv"

lifesat = pd.read_csv(data_root) #leída desde el csv
X = lifesat[["GDP per capita (USD)"]].values # tomo los valores de x=gpd per capita
y = lifesat[["Life satisfaction"]].values # tomo los valores de y= satisfaccion

# Acá elijo un modelo de regresión lineal
# model = LinearRegression() 
model = KNeighborsRegressor(n_neighbors=3)
# Train the model
model.fit(X, y)

# Predicción para Chipre
X_new = [[37_655.2]] # Cyprus' GDP per capita in 2020
model.predict(X_new) # output: [[6.30165767]]

# Ploteo
lifesat.plot(kind='scatter', grid=True,
x="GDP per capita (USD)", y="Life satisfaction") # metodo built in de ploteo de df
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.axis([23_500, 62_500, 4, 9])
plt.show()
