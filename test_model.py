from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import mlp_model
import alexnet_model
import vgg_model
from detection_boxes import disnet
from torch.nn.functional import mse_loss
from mlp_model import split_dataset, CustomDataset, transform
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import logging
import json

logging.basicConfig(
        filename ='test_model.log', 
        level=logging.INFO, 
        filemode='w',
        format='%(asctime)s::%(levelname)s::%(filename)s::%(lineno)d::%(message)s')

logger = logging.getLogger(__name__)


dataset_path = 'dataset_cleaned'
dataset_dir = 'data/dataset_cleaned'
# dataset_disnet_dir = 'detection_boxes'
dataset_disnet_dir = 'detection_boxes/data_midas_large_v3'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = alexnet_model.AlexNet()
model = vgg_model.VGGNet()
# model = mlp_model.MLP()
model = disnet.DisNet(input_size=39)
# model.load_state_dict(torch.load(r'checkpoints/mlp_v3/model.pth'))
model.load_state_dict(torch.load(r'checkpoints/disnet_v4/model.pth'))


# Carregando dados de teste
# Dividindo os dados
def teste_dados_dataset(dir, image_names_test, targets_test, log=False):

    if log == True:
        print('teste')
        ass_img = image_names_test[30]
        print(ass_img)

        ass_target = targets_test[30]
        print(ass_target)

        check = float(open(f"{dir}/{ass_img.split('.')[0]}.txt").readline().strip())
        print(check)

    for i in range(len(image_names_test)):
        ass_img = image_names_test[i]
        ass_target = targets_test[i]
        check = float(open(f"{dir}/{ass_img.split('.')[0]}.txt").readline().strip())
        tolerance = 1e-6  # Defina a margem de tolerância adequada
        assert abs(ass_target - check) < tolerance
    
# Função para calcular o erro médio em %
def calculate_percentage_error(actual, predicted):
    return ((predicted - actual) / actual) * 100

# Função para calcular a saída do modelo em N amostras de entrada
def eval_model(model, dataset_test, n_samples=None):

    np.random.seed(42)
    if n_samples:
        random_indices = np.random.choice(len(dataset_test), n_samples, replace=False)
        indices = random_indices
    else:
        all_indices = range(len(dataset_test))
        indices = all_indices

    # Listas para armazenar os resultados
    target_distances = []
    predicted_distances = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            img, target = dataset_test[idx]
            output = model(img.unsqueeze(0)) # Adiciona uma dimensão para o batch
            predicted_distance = output.item()
            target_distances.append(target.item())
            predicted_distances.append(predicted_distance)

    if n_samples:
        return target_distances, predicted_distances, indices
    
    return target_distances, predicted_distances



def main(dataset_dir,  dataset_disnet_dir, transform, model):
    

    # Carregando os dados de teste
    # img_names_train, targets_train, img_names_val, targets_val, img_names_test, targets_test = split_dataset(dataset_path)
    # teste_dados_dataset(dataset_path, img_names_test, targets_test)
    # Carregando dados modelo padrão
    dataset_test = CustomDataset(os.path.join(dataset_dir, 'test'), transform)
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False) # Não estou usando

    
    # Carregando dados DisNet
    X_test, y_test = disnet.load_data(filepath=os.path.join(dataset_disnet_dir, 'test_disnet.csv'), midas=True)
    dataset_test = TensorDataset(X_test, y_test)
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)
    
    
    target_distances, predicted_distances = eval_model(model, dataset_test)
    

    #### Avaliação no conjunto de teste ####
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for imgs, targets in dataloader_test:
            outputs = model(imgs)
            loss = mse_loss(outputs, targets) # reshape(-1, 1) -> shape (4,1)
            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    test_mse = running_loss / total_samples # MSE global
    test_rmse = np.sqrt(test_mse) # RMSE global


    # print(predicted_distances)
    # print(outputs_list)


    # Métricas nos dados de teste
    mse_test = np.mean((np.array(target_distances) - np.array(predicted_distances))**2)
    rmse_test = np.sqrt(mse_test)
    mae_test = np.mean(np.abs(np.array(target_distances) - np.array(predicted_distances)))
    mape_test = np.mean(np.abs(np.array(target_distances) - np.array(predicted_distances)) / np.array(target_distances)) * 100
    std_test = np.std(np.abs(np.array(target_distances) - np.array(predicted_distances)))
    r_squared_test = r2_score(target_distances, predicted_distances)


    # Métrcas na validação e treinamento do modelo
    with open('metrics.json', 'r') as f:
        dados = json.load(f)


    #rmse_val = dados['validacao']['rmses'][-1]
    #rmse_train = dados['treinamento']['rmses'][-1]


    mse_val = dados['validacao']['val_losses']
    rmse_val = dados['validacao']['val_rmses']
    mse_train = dados['treinamento']['train_losses']
    rmse_train = dados['treinamento']['train_rmses']


    # Calcular R² e os parâmetros da reta
    regression = LinearRegression()
    target_distances = np.array(target_distances).reshape(-1, 1)
    predicted_distances = np.array(predicted_distances).reshape(-1, 1)
    regression.fit(target_distances, predicted_distances)
    r_squared = r2_score(target_distances, predicted_distances)
    slope = regression.coef_[0][0]
    intercept = regression.intercept_

    logger.info(f"Coeficiente de Determinação (R²): {r_squared:.4f}")
    logger.info(f"Parâmetros da Reta: Coeficiente Angular (Slope): {slope:.4f}, Intercepto: {np.float32(intercept.item()):.4f}")


    # Salvar as métricas
    logger.info('-------------Avaliação nas amostras--------------')
    logger.info('mse_test: %s', mse_test)
    logger.info('rmse_test: %s', rmse_test)
    logger.info('mae_test: %s', mae_test)
    logger.info('mape_test: %s', mape_test)
    logger.info('std_test: %s', std_test)
    logger.info('r_squared_test: %s', r_squared_test)

    logger.info('------------------Avaliação nos dados de origem-----------')    
    logger.info('mse_test_true: %s', test_mse)
    logger.info('rmse_test_true: %s', test_rmse)

    logger.info('mse_val: %s', mse_val[-1])
    logger.info('rmse_val: %s', rmse_val[-1])

    logger.info('mse_train: %s', mse_train[-1])
    logger.info('rmse_train: %s', rmse_train[-1])


    # Gráfico de regressão do valor real pelo predito
    plt.figure(figsize=(10, 5))
    plt.scatter(target_distances, predicted_distances, color='blue')
    plt.title('Valor Real pelo Predito')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predito')
    plt.grid(True)
    # Adiciona a linha de regressão ao gráfico
    plt.plot(target_distances, regression.predict(target_distances), color='green')
    # Adiciona linha de regressão ideal
    plt.plot(target_distances, target_distances, color='red')
    plt.legend(['Pontos', 'Regressão Linear', 'Reta Ideal'])
    # Adicionando a equação da reta e o valor de R² ao gráfico
    equation = f'Reta: y = {slope:.4f}x + {np.float32(intercept.item()):.4f}\nR² = {r_squared:.4f}'
    plt.text(0.8, 0.5, equation, ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('regression_plot.png')
    plt.show()
    
    

    # Métricas durante o treinamento
    plt.figure(figsize=(10, 5))
    plt.plot(mse_train, label='Treinamento')
    plt.plot(mse_val, label='Validação')
    plt.title('Função de Perda durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


    plt.plot(rmse_train, label='Treinamento')
    plt.plot(rmse_val, label='Validação')
    plt.title('RMSE durante o Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('rmse.png')
    plt.show()




    #TODO - Implementar a função de plotar a imagem com a distância real e a distância predita

    """
    img = img_names_test[237]
    img_path = f"{dataset_path}/{img}"
    logger.info('%s',img)
    img_pil = Image.open(img_path)
    distance = open(f"{dataset_path}/{img.replace('png', 'txt')}").readline().strip()
    logger.info('%s',distance)


    plt.figure(figsize=(10, 5))
    plt.imshow(img_pil)
    plt.title(f'image_path: {img_path}, distance real:{distance}')
    plt.savefig('test_image')
    plt.show()
    """

if __name__ == '__main__':
    main(dataset_dir=dataset_dir, dataset_disnet_dir=dataset_disnet_dir, transform=transform, model=model)
