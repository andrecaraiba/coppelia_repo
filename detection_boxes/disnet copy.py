import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import mse_loss
import json


num_epochs = 200


class DisNetDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df_dataset = pd.read_csv(csv_file)
        self.transform = transform


    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        # feature = self.df_dataset.loc[idx, ['1/Bh', '1/Bw', '1/Bd']].values
        # target = self.df_dataset.loc[idx, 'Target'].values

        feature = self.df_dataset.loc[idx, ['1/Bh', '1/Bw', '1/Bd']]
        target = self.df_dataset.loc[idx, 'Target']

        print('type(feature):', type(feature))
        print('type(target):', type(target))
        print('feature:', feature)
        print('target:', target)

        if self.transform:
            feature = self.transform.fit_transform(feature)
            target = self.tranform.fit_transform(target)

            feature = torch.tensor(feature, dtype=torch.float32)
            target = torch.tensor([target], dtype=torch.float32)

        return feature, target

class DisNet(nn.Module):
    def __init__(self):
        super(DisNet, self).__init__()
        self.fc1 = nn.Linear(3, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_data(filepath):
    data = pd.read_csv(filepath)
    features = data[['1/Bh', '1/Bw', '1/Bd']].values
    targets = data['Target'].values

    # normalizandos os dados
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Convertendo para tensores
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    return features, targets

def main():
    filepath = 'bounding_boxes_with_targets.csv'

    
    features, target = load_data(filepath)
    
    # Normalizando os dados
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Convertendo para tensores
    features = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32).view(-1, 1)
    
    # Dividindo os dados em conjuntos de treino, validação e teste
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Criando DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Construindo o modelo
    model = DisNet()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Listas para armazenar métricas
    train_losses = []
    train_rmses = []
    val_losses = []
    val_rmses = []
    
    # Treinando o modelo
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_rmses = []
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            rmse = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_rmses.append(rmse.item())
        
        train_losses.append(np.mean(epoch_losses))
        train_rmses.append(np.mean(epoch_rmses))
        
        # Validação
        model.eval()
        val_epoch_losses = []
        val_epoch_rmses = []
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                rmse = torch.sqrt(loss)
                val_epoch_losses.append(loss.item())
                val_epoch_rmses.append(rmse.item())
        
        val_losses.append(np.mean(val_epoch_losses))
        val_rmses.append(np.mean(val_epoch_rmses))
        model.train()  # Reativar o modo de treinamento para a próxima época
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Train RMSE: {train_rmses[-1]}, Val Loss: {val_losses[-1]}, Val RMSE: {val_rmses[-1]}')

    # Salvar as métricas em um arquivo JSON
    metrics = {
        'treinamento': {
            'losses': train_losses,
            'rmses': train_rmses
        },
        'validacao': {
            'losses': val_losses,
            'rmses': val_rmses
        }
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Salvar o modelo 
    torch.save(model.state_dict(), 'model.pth')

    # Carregar o modelo
    model.load_state_dict(torch.load('model.pth'))


    # Teste final
    model.eval()
    test_losses = []
    test_rmses = []
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            rmse = torch.sqrt(loss)
            test_losses.append(loss.item())
            test_rmses.append(rmse.item())
    print(f'Final Test Loss: {np.mean(test_losses)}, Final Test RMSE: {np.mean(test_rmses)}')

    # Plotando gráficos
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Treinamento')
    plt.plot(val_losses, label='Validação')
    plt.title('Função de Perda durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


    plt.plot(train_rmses, label='Treinamento')
    plt.plot(val_rmses, label='Validação')
    plt.title('RMSE durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('rmse.png')
    plt.show()

if __name__ == "__main__":
    main()
