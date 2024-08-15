import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

# Definir hiperparametros
learning_rate = 1e-3
num_epochs = 150
dataset_path = r'dataset_cleaned'
data_dir = r'data/dataset_cleaned'

# Verificar disponibilidade de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_dataset(img_dir, split_train=0.7, split_val=0.15, split_test=0.15):
    img_names = [img for img in os.listdir(img_dir) if img.endswith('.png')]
    img_names.sort()  # Garante que está na mesma ordem que os targets
    targets = [float(open(f"{img_dir}/{img_name.split('.')[0]}.txt").readline().strip()) for img_name in img_names]
    total_imgs = len(img_names)
    train_end = int(total_imgs * split_train)
    val_end = train_end + int(total_imgs * split_val)

    img_names_train = img_names[:train_end]
    targets_train = targets[:train_end]

    img_names_val = img_names[train_end:val_end]
    targets_val = targets[train_end:val_end]

    img_names_test = img_names[val_end:]
    targets_test = targets[val_end:]

    return img_names_train, targets_train, img_names_val, targets_val, img_names_test, targets_test



class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = []
        self.targets = []

        files = os.listdir(data_dir)
        files.sort()

        for file in files:
            if file.endswith('.txt'):
                target_path = os.path.join(data_dir, file)
                target = float(open(target_path).readline().strip())
                self.targets.append(target)
            
            if file.endswith('.png'):
                self.img_names.append(file)
                

        assert len(self.img_names) == len(self.targets)
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.data_dir, self.img_names[idx])
        image = Image.open(img_path)
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)
            target = torch.tensor([self.targets[idx]])
            
        return image, target
    
# Transformações (normalização e outras necessárias)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Redimensiona a imagem
    transforms.ToTensor(),  # Converte a imagem para tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.3, 0.3, 0.3]),  # Normaliza
])

# Definindo a rede neural (VGG 16)
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x


def main():
    # Dividindo os dados
    # img_names_train, targets_train, img_names_val, targets_val, img_names_test, targets_test = split_dataset(dataset_path)

    # Carrega os datasets
    dataset_train = CustomDataset(os.path.join(data_dir, 'train'), transform)
    dataset_val = CustomDataset(os.path.join(data_dir, 'val'), transform)
    dataset_test = CustomDataset(os.path.join(data_dir, 'test'), transform)

    # Criando os Dataloaders carregar os dados em batch
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

    # Debug Inputs
    for imgs, targets in dataloader_train:
        print('imgs: ', imgs.dtype)
        print('labels: ', targets.dtype)
        break

    # Instanciando a rede e definindo o otimizador e a função de perda
    model = VGGNet().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history  = {'loss': [], 'val_loss': [], 'rmse': [], 'val_rmse': []}
    # Loop de treinamento com tqdm
    for epoch in range(num_epochs):

        ### Trainning ###
        model.train()
        running_loss = 0.0
        with tqdm(dataloader_train, unit="batch") as tepoch:
            for imgs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                imgs = imgs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(imgs)
                # print('outputs: ', outputs.shape)
                # print('targets: ', targets.shape)
                loss = criterion(outputs, targets.view(-1, 1))
                
                # Backward pass e otimização
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                
                # Atualizando a barra de progresso
                tepoch.set_postfix(loss=loss.item())
                
        
        # Calculando médias das métricas da época e armazenando
        epoch_loss = running_loss / len(dataloader_train.dataset)
        epoch_rmse = np.sqrt(epoch_loss)
        history['loss'].append(epoch_loss)
        history['rmse'].append(epoch_rmse)
        print(f"Final Epoch {epoch+1} - Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}")
    
        
        ### Validation ###
        model.eval()  # Coloca o modelo em modo de avaliação
        running_loss = 0.0
        with torch.no_grad():  # Desativa o cálculo de gradiente para economizar memória e computação
            with tqdm(dataloader_val, unit="batch") as tepoch:  
                for imgs, targets in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}")
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    # Forward pass
                    outputs = model(imgs)
                    loss = criterion(outputs, targets.view(-1, 1))
                    running_loss  += loss.item() * imgs.size(0)

            epoch_val_loss = running_loss / len(dataloader_val.dataset)
            epoch_val_rmse = np.sqrt(epoch_val_loss)
            history['val_loss'].append(epoch_val_loss)
            history['val_rmse'].append(epoch_val_rmse)
            print(f"Epoch {epoch+1} - Loss Validation: {epoch_val_loss:.4f}, RMSE de Validation: {epoch_val_rmse:.4f}")

    #Salvar as métricas em JSON
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

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)


    # Salvar o modelo
    torch.save(model.state_dict(), "model.pth")


    # Carregar o modelo
    model.load_state_dict(torch.load("model.pth"))

    #### Avaliação no conjunto de teste ####
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, targets in dataloader_test:
            outputs = model(imgs.to(device))
            loss = criterion(outputs, targets.to(device).view(-1, 1))
            running_loss += loss.item() * imgs.size(0)
        
    test_mse = running_loss / len(dataloader_test.dataset) # MSE global
    test_rmse = np.sqrt(test_mse) # RMSE global
    print(f"Perda no Teste: {test_mse:.4f}, RMSE no Teste: {test_rmse:.4f}")

    # Gráfico para a função de perda
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Treinamento')
    plt.plot(history['val_loss'], label='Validação')
    plt.title('Função de Perda durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    # salva o gráfico
    plt.savefig('loss.png')
    plt.show()

    # Gráfico para o RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(history['rmse'], label='Treinamento')
    plt.plot(history['val_rmse'], label='Validação')
    plt.title('RMSE durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.legend()
    # salva o gráfico
    plt.savefig('rmse.png')
    plt.show()

if __name__ == '__main__':
    main()




