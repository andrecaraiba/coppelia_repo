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


num_epochs = 250
midas = True


class DisNetDataset(Dataset): # Não to usando
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
    def __init__(self, input_size):
        super(DisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_data(filepath, midas=False):
    data = pd.read_csv(filepath)
    features = data[['1/Bh', '1/Bw', '1/Bd']].values
    targets = data['Target'].values

    if midas:
        # features = data[['1/Bh', '1/Bw', '1/Bd', 'midas_max', 'midas_max_relative']].values
        features = data.iloc[:, np.r_[5, 6, 7, 14:data.shape[1]]].values
        targets = data['Target'].values

    # normalizandos os dados
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Convertendo para tensores
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    return features, targets

def main():
    # filepath = 'bounding_boxes_with_targets.csv'
    root_filepath = r'detection_boxes/data_midas_large_v3'
    input_size = 39
    if midas == False:
        root_filepath = r'detection_boxes'
        input_size = 3

    # dataset = DisNetDataset(filepath, transform=StandardScaler())
    X_train, y_train = load_data(filepath=os.path.join(root_filepath, 'train_disnet.csv'), midas=midas)
    X_val, y_val = load_data(filepath=os.path.join(root_filepath, 'val_disnet.csv'), midas=midas)
    X_test, y_test = load_data(filepath=os.path.join(root_filepath, 'test_disnet.csv'), midas=midas)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Criando DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Construindo o modelo
    model = DisNet(input_size)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    history = {'loss': [], 'val_loss': [], 'rmse': [], 'val_rmse': []}
    for epoch in range(num_epochs):

        ### Training ###
        model.train()
        running_loss = 0.0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_rmse = np.sqrt(epoch_loss)
        history['loss'].append(epoch_loss)
        history['rmse'].append(epoch_rmse)

        ### Validation ###
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * data.size(0)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_rmse = np.sqrt(epoch_val_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_rmse'].append(epoch_val_rmse)        
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train RMSE: {epoch_rmse:.4f}, Val Loss: {epoch_val_loss:.4f}, Val RMSE: {epoch_val_rmse:.4f}')


    # Salvar as métricas em JSON
    metrics = {
        "treinamento": {
            "train_losses": history['loss'], 
            "train_rmses": history['rmse']
        },
        "validacao": {
            "val_losses": history['val_loss'], 
            "val_rmses": history['val_rmse']
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
    running_loss = 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * data.size(0)
        test_mse = running_loss / len(test_loader.dataset)
        test_rmse = np.sqrt(test_mse)
    print(f'Final Test Loss: {test_mse:.4f}, Final Test RMSE: {test_rmse:.4f}')

    # Plotando gráficos
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Treinamento')
    plt.plot(history['val_loss'], label='Validação')
    plt.title('Função de Perda durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


    plt.plot(history['rmse'], label='Treinamento')
    plt.plot(history['val_rmse'], label='Validação')
    plt.title('RMSE durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('rmse.png')
    plt.show()

if __name__ == "__main__":
    main()
