from typing import List, Dict
import math
import csv
import os
from ultralytics import YOLO
import torch
from PIL import Image

# Constantes globais
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
OBJECT_MEAN_DIMENSIONS = (0.1, 0.1, 0.1)  # Ch, Cw, Cb in meters
DIR_PATH_DATASET = r'dataset' # Não está sendo usado

def load_images(directory_path: str) -> List[str]:
    """Carrega todas as imagens PNG de um diretório especificado."""
    images = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.png')]
    return images

def load_targets(directory_path: str) -> Dict[str, int]:
    """Carrega os targets de cada arquivo .txt em um dicionário."""
    targets = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            image_name = filename.replace('.txt', '')
            with open(os.path.join(directory_path, filename), 'r') as file:
                target = float(file.read().strip())
                targets[image_name] = target
    return targets

def calculate_additional_features(box: List[float]) -> List[float]:
    """Calcula características adicionais para bounding boxes."""
    x, y, w, h = box
    Bh = h / IMAGE_HEIGHT
    Bw = w / IMAGE_WIDTH
    Bd = math.sqrt(w**2 + h**2) / math.sqrt(IMAGE_WIDTH**2 + IMAGE_HEIGHT**2)
    
    Ch, Cw, Cb = OBJECT_MEAN_DIMENSIONS
    return [x, y, w, h, 1/Bh, 1/Bw, 1/Bd, Ch, Cw, Cb]

def extract_features_and_targets(model, images: List[str], targets: Dict[str, int]) -> Dict[str, List[List[float]]]:
    """Extrai características de bounding box das imagens e inclui os targets."""
    all_bounding_boxes = {}
    
    for image in images:
        result = next(model([image], stream=True))
        boxes = result.boxes.xywh
        image_name = os.path.basename(result.path).replace('.png', '')
        
        bounding_boxes = boxes.tolist()
        extended_features = [calculate_additional_features(box) for box in bounding_boxes]
        extended_features_with_target = [features + [targets[image_name]] for features in extended_features]
        
        all_bounding_boxes[image_name] = extended_features_with_target
        
        del result
        torch.cuda.empty_cache()
    
    return all_bounding_boxes

def save_features_to_csv(bounding_boxes: Dict[str, List[List[float]]], output_path: str) -> None:
    """Salva as características extraídas e os targets em um arquivo CSV."""
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "x", "y", "w", "h", "1/Bh", "1/Bw", "1/Bd", "Ch", "Cw", "Cb", "Target"])
        
        for image_name, boxes in bounding_boxes.items():
            for box in boxes:
                writer.writerow([image_name] + box)

def main(data_dir: str, save_path_file: str) -> None:
    """Função principal que executa o processo de extração de características e inclui os targets."""
    model_path = '/home/dreca/coppelia_repo/detection_boxes/runs/detect/train3/weights/best.pt'
    model = YOLO(model_path)
    images = load_images(data_dir)
    targets = load_targets(data_dir)
    bounding_boxes = extract_features_and_targets(model, images, targets)
    save_features_to_csv(bounding_boxes, save_path_file)

if __name__ == "__main__":
    root_dataset_dir = r'data/dataset_cleaned'
    save_dir = r'detection_boxes'

    for dataset in os.listdir(root_dataset_dir):
        main(data_dir=os.path.join(root_dataset_dir, dataset), 
             save_path_file=os.path.join(save_dir, dataset + '_disnet.csv'))







