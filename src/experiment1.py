import mlflow
import mlflow.pytorch
import mlflow.onnx

import onnx

import joblib

import torch
import torch.onnx
from torch import nn
from torch import optim

from NeuralNetwork import MyIrisNet

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

params_exp = {
    "test_size": 0.2,
    "learning_rate": 0.01,
    "epoch_count": 100
}

mlflow.set_tracking_uri("http://localhost:5000")

scaler = StandardScaler()
iris = load_iris()
iris_X1, iris_Y = iris.data, iris.target
iris_X = scaler.fit_transform(iris_X1)
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=params_exp["test_size"])

X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.LongTensor(Y_train)
X_test_tensor = torch.FloatTensor(X_test)
Y_test_tensor = torch.LongTensor(Y_test)

model = MyIrisNet()

criterion = nn.CrossEntropyLoss()
optimazer = optim.Adam(model.parameters(), lr=params_exp["learning_rate"])

mlflow.set_experiment("iris-classification_1")
with mlflow.start_run():
    mlflow.log_params({
        "input_dim": 4,
        "hidden_dim": 50,
        "output_dim": 3,
        "activation": "ReLU",
        "optimizer": "Adam",
        "criterion": "CrossEntropyLoss",
        "scaler": "StandardScaler",
        **params_exp
    })

    for epoch in range(params_exp["epoch_count"]):
        optimazer.zero_grad()
        outputs = model.forward(X_train_tensor)

        loss = criterion(outputs, Y_train_tensor)
        
        loss.backward()
        optimazer.step()

    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        _, y_pred = torch.max(y_pred_tensor, 1)
    
    mlflow.log_metrics({
        "accuracy_score": float(accuracy_score(Y_test, y_pred)),
        "precision_score": float(precision_score(Y_test, y_pred, average='weighted')),
        "recall_score": float(recall_score(Y_test, y_pred, average='weighted')),
        "f1_score": float(f1_score(Y_test, y_pred, average='weighted')),
    })

    dummy_input = torch.randn(1, 4) 

    torch.onnx.export(
        model,                      # ваша обученная модель
        dummy_input,                # пример входных данных
        "model.onnx",               # название файла
        export_params=True,         # сохраняем веса внутри файла
        opset_version=11,           # стандартная версия ONNX
        do_constant_folding=True,   # оптимизация
        input_names=['input'],      # имя входа для API
        output_names=['output'],    # имя выхода для API
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # разрешаем разный размер батча
    )

    onnx_model = onnx.load("model.onnx")

    # Логируем в MLflow
    mlflow.onnx.log_model(
        onnx_model=onnx_model,
        artifact_path="onnx-model",
        registered_model_name="iris_onnx_model"
    )

    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")

    # mlflow.pytorch.log_model(model, "model", registered_model_name="iris_classification")

