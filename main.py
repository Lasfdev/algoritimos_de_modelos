# %%
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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
# Modelo KNN
with mlflow.start_run(run_name="KNN_Baseline"):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))

    mlflow.log_param("modelo","KNN")
    mlflow.log_param("n_neighbors",5)
    mlflow.log_metric("accuracy", acc_knn)
    mlflow.sklearn.log_model(knn, "modelo_knn")

# %%
# modelo SVM 
with mlflow.start_run(run_name="SVM_Baseline"):
    svm = SVC(C=1.0, kernel="rbf")
    svm.fit(X_train,y_train)
    acc_svm = accuracy_score(y_test, svm.predict(X_test))

    mlflow.log_param("modelo","SVM")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("kernel","rbf")
    mlflow.log_metric("accuracy", acc_svm)
    mlflow.sklearn.log_model(svm, "modelo_svm")

# %%

# Supondo que você já calculou acc_knn e acc_svm
modelos = ['KNN', 'SVM']
acuracias = [acc_knn, acc_svm]

# Cria o gráfico de barras
fig, ax = plt.subplots(figsize=(6, 4))
barras = ax.bar(modelos, acuracias, color=['#4C72B0', '#55A868'], width=0.5)

# Configurações visuais
ax.set_ylim(0, 1.1)  # Escala de 0 a 110% para caber o texto no topo
ax.set_ylabel('Acurácia')
ax.set_title('Comparação de Acurácia: KNN vs SVM')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Adiciona o valor exato em cima de cada barra
for barra in barras:
    altura = barra.get_height()
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        altura + 0.02,
        f'{altura:.2%}',
        ha='center',
        fontweight='bold'
    )

plt.tight_layout()
plt.show()

# Opcional: Salva o gráfico dentro de uma run no MLflow
with mlflow.start_run(run_name="Comparacao_Final"):
    mlflow.log_figure(fig, "comparacao_acuracia.png")
