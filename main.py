# %%
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# %%
path = "./data/processed/dataset_features.npz"

# %%
with np.load(path) as dados:
    X = dados['X']
    y = dados['y']

# %%
print(y)
print(X)

# %%
# Definindo o nome do experimento 
mlflow.set_experiment("classificacao_frutas")

# %%
# inicializar o codificador
le = LabelEncoder()

# tranformando os labels em numeros
y = le.fit_transform(y)

# print(le.classes_)  // ver qual numero corresponde a qual classe

# %%
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)

# %%


# %%
# depois treinar os modelos 
# metricas e hiperparametrpos
# mostras resultados de comparação no grafico