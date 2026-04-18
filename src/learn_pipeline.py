import torch
from torch import nn
from torch import optim

from NeuralNetwork import MyIrisNet

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

scaler = StandardScaler()
iris = load_iris()
iris_X1, iris_Y = iris.data, iris.target
iris_X = scaler.fit_transform(iris_X1)
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.2)

X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.LongTensor(Y_train)
X_test_tensor = torch.FloatTensor(X_test)
Y_test_tensor = torch.LongTensor(Y_test)

model = MyIrisNet()

criterion = nn.CrossEntropyLoss()
optimazer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimazer.zero_grad()
    outputs = model.forward(X_train_tensor)

    loss = criterion(outputs, Y_train_tensor)
    
    loss.backward()
    optimazer.step()

with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    _, y_pred = torch.max(y_pred_tensor, 1)
    
print(classification_report(Y_test, y_pred))

# torch.save(model.state_dict(), "model_state.pth")
