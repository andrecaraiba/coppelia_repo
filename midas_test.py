import cv2
import torch
import urllib.request
from ultralytics import YOLO
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import math

filename = r'data/dataset_cleaned/test/image758.png'
filename_2 = r'data/dataset_cleaned/test/image88.png'
filename_3 = r'data/dataset_cleaned/test/image459.png'
model_path = r'/home/dreca/coppelia_repo/detection_boxes/runs/detect/train3/weights/best.pt'
img_not_found = []

# Constantes globais
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
OBJECT_MEAN_DIMENSIONS = (0.1, 0.1, 0.1)  # Ch, Cw, Cb in meters


def load_data(data_dir, yolo_model_path, midas_model, midas_transform):
    i = 0

    data_dict = {'Image': [],
                 'x': [], 
                 'y': [],
                 'w': [],
                 'h': [],
                 '1/Bh': [],
                 '1/Bw': [],
                 '1/Bd': [],
                 'Ch': [],
                 'Cw': [],
                 'Cb': [],
                 'midas_max': [],
                 'midas_max_relative': [],
                 'Target': [],}
    
    # create columns for dept_features
    for idx in range(36):
        data_dict[f'dpt_{idx+1}'] = []
    
    
    
    yolov8 = YOLO(yolo_model_path)

    files = os.listdir(data_dir)
    files.sort()
    png_files = [file for file in files if file.endswith('.png')]
    txt_files = [file for file in files if file.endswith('.txt')]

    files_list = list(zip(png_files, txt_files))

    for file in files_list:
        img_file = file[0]
        txt_file = file[1]

        # img_path
        img_path = os.path.join(data_dir, img_file)


        # midas
        midas_max, midas_max_relative, dept_features, yolo_features = extract_MiDaS_feature(i, yolov8, img_path, midas_model, midas_transform)
        i += 1
        
        if midas_max is None:
            continue

        # target
        with open(os.path.join(data_dir, txt_file), 'r') as file:
            target = float(file.readline().strip())
        
        data_dict['Image'].append(os.path.basename(img_path).replace('.png', ''))
        data_dict['x'].append(yolo_features['x'])
        data_dict['y'].append(yolo_features['y'])
        data_dict['w'].append(yolo_features['w'])
        data_dict['h'].append(yolo_features['h'])
        data_dict['1/Bh'].append(yolo_features['1/Bh'])
        data_dict['1/Bw'].append(yolo_features['1/Bw'])
        data_dict['1/Bd'].append(yolo_features['1/Bd'])
        data_dict['Ch'].append(OBJECT_MEAN_DIMENSIONS[0])
        data_dict['Cw'].append(OBJECT_MEAN_DIMENSIONS[1])
        data_dict['Cb'].append(OBJECT_MEAN_DIMENSIONS[2])
        data_dict['midas_max'].append(midas_max)
        data_dict['midas_max_relative'].append(midas_max_relative)
        for idx, feature in enumerate(dept_features):
            data_dict[f'dpt_{idx+1}'].append(feature)
        data_dict['Target'].append(target)

        load_step = len(files_list) // 100
        load = i // load_step
        print('['+load*'='+'>'+(100-load)*' '+']', f'{i}/{len(files_list)}', end='\r')

    return data_dict


def extract_MiDaS_feature(i, yolov8, img_path, midas, midas_transform):
    # passar para o device GPU
    # img_path = r'data/dataset_cleaned/train/image328.png'
    for image in [img_path]:
        result = next(yolov8([image], stream=True))
        if len(result.boxes.xyxy) == 0:
            print(f'Not found box in image {i}...{img_path}')
            img_not_found.append(img_path)

            return None, None, None, None
        
        boxes_xyxy = result.boxes.xyxy
        # image_name = os.path.basename(result.path).replace('.png', '')
        bounding_boxes_xyxy = boxes_xyxy.tolist()
        # print('bounding_boxes:', bounding_boxes)

        plot = result.plot(line_width=1, labels=True)
        plt.imshow(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))
        plt.show()
        # cv_image = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
        # print(cv_image[128:137, 122:134])
        # plt.imshow(cv_image[125:139, 120:136])
        # plt.show()

        # Yolo
        yolo_features = {}
        boxes_xywh = result.boxes.xywh
        bounding_boxes_xywh = boxes_xywh.tolist()
        for box in bounding_boxes_xywh:
            x, y, w, h = box
            Bh = h/ IMAGE_HEIGHT
            Bw = w/ IMAGE_WIDTH
            Bd = math.sqrt(w**2 + h**2) / math.sqrt(IMAGE_WIDTH**2 + IMAGE_HEIGHT**2)

            yolo_features['x'] = x
            yolo_features['y'] = y
            yolo_features['w'] = w
            yolo_features['h'] = h
            yolo_features['1/Bh'] = Bh
            yolo_features['1/Bw'] = Bw
            yolo_features['1/Bd'] = Bd

        del result
        torch.cuda.empty_cache()


    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = midas_transform(img).to(device)

    # Predict and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), # Adiciona uma dimensão com tamanho 1 pq a função espera um tensor 4D (mini-batch, channels, height, width)
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze() # remove todas as dimensões com tamanho 1

    output = prediction.cpu().numpy() # Converte para numpy ndarray

    # carrega a imagem original
    print(img_path)
    plt.imshow(img) # Carrega a imagem original
    plt.show()

    # carrega a imagem de profundidade
    plt.imshow(output)
    plt.show()


    # draw bounding boxes
    for box in bounding_boxes_xyxy:
        x1, y1, x2, y2 = box
        box_color = int(output.max())
        print(box_color)
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (box_color), 1)

    # Show result
    # print(int(x1), int(x2), int(y1), int(y2))
    # print('output:', output[128:137, 122:134])
    # rect = output[int(y1):int(y2+1), int(x1):int(x2+1)] # select the region of interest
    rect = output[int(y1+1):int(y2), int(x1+1):int(x2)] # select the region of interest intern
    max_value = rect.max()
    max_relative = (max_value - output.min()) / (output.max() - output.min()) 

    dept_image = cv2.resize(rect, (6, 6), interpolation=cv2.INTER_CUBIC)
    dept_features = dept_image.flatten()
    norm_dept_features = (dept_features - output.min())/ (output.max() - output.min()) 

    # plt.imshow(rect) # Carrega os pixels dentro da box
    # plt.show()
    print(output)

    # Carrega a imagem de profundidade com as bounding boxes
    plt.imshow(output)
    plt.show()
    return max_value, max_relative, norm_dept_features, yolo_features
    

if __name__ == '__main__':


    root_dataset_dir = r'data/dataset_cleaned'
    yolo_model_path = r'/home/dreca/coppelia_repo/detection_boxes/runs/detect/train3/weights/best.pt'


     # -------------------------- MiDaS --------------------------
    model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)

    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


    data_train = load_data(data_dir=os.path.join(root_dataset_dir, 'train'),
                     yolo_model_path=yolo_model_path,
                     midas_model=midas, midas_transform=transform)
    
    data_val = load_data(data_dir=os.path.join(root_dataset_dir, 'val'),
                     yolo_model_path=yolo_model_path,
                     midas_model=midas, midas_transform=transform)
    
    data_test = load_data(data_dir=os.path.join(root_dataset_dir, 'test'),
                     yolo_model_path=yolo_model_path,
                     midas_model=midas, midas_transform=transform)

    pd.DataFrame(data_train).to_csv('train_disnet.csv', index=False)
    pd.DataFrame(data_val).to_csv('val_disnet.csv', index=False)
    pd.DataFrame(data_test).to_csv('test_disnet.csv', index=False)
    print(img_not_found)