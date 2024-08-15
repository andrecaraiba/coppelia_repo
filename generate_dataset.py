import os
import random
import shutil
from typing import List


class GenerateDataset:
    """
    Essa classe gera o dataset a partir de um diretório contendo as imagens e seus respectivos targets.
    params:
        source_dir: str - O diretório onde estão as imagens e os targets.
    """
    
    def __init__(self, source_dir: str, train_size=0.7, val_size=0.15):
        self.source_dir = source_dir
        self.train_size = train_size
        self.val_size = val_size
        self.total_files = 0

    def generate_dataset(self) -> None:

        # Cria os diretórios train, val e test
        self.create_dirs()

        # Separa os arquivos em train, val e test
        train_files, val_files, test_files = self.split_dataset()

        # Move os arquivos para os diretórios train, val e test
        target_dirs = [os.path.join(self.source_dir, target) for target in ['train', 'val', 'test']]
        print(f'''movendo total {self.total_files} imagens com seus respectivos targets... {len(train_files)} arquivos para train, {len(val_files)} arquivos para val e {len(test_files)} arquivos para test''')
        
        self.move_files(train_files, target_dirs[0])
        self.move_files(val_files, target_dirs[1])
        self.move_files(test_files, target_dirs[2])

    def create_dirs(self) -> None:
        train_dir = os.path.join(self.source_dir, 'train')
        val_dir = os.path.join(self.source_dir, 'val')
        test_dir = os.path.join(self.source_dir, 'test')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

    def split_dataset(self) -> List[str]:
        files = os.listdir(self.source_dir)
        files.sort()
        png_files = [file for file in files if file.endswith('.png')]
        txt_files = [file for file in files if file.endswith('.txt')]

        files_list = list(zip(png_files, txt_files))
        
        random.shuffle(files_list, random=random.seed(42))
        
        total_files = len(files_list)
        self.total_files = total_files

        train_end = int(total_files * self.train_size)
        val_end = train_end + int(total_files * self.val_size)

        train_files = files_list[:train_end]
        val_files = files_list[train_end:val_end]
        test_files = files_list[val_end:]


        return train_files, val_files, test_files
    


    def move_files(self, files: List[str], target_dir: str) -> None:

        for file in files:
            img_file = file[0]
            txt_file = file[1]

            # move img file
            source_path = os.path.join(self.source_dir, img_file)
            dest_path = os.path.join(target_dir, img_file)
            shutil.move(source_path, dest_path)

            # move txt file
            source_path = os.path.join(self.source_dir, txt_file)
            dest_path = os.path.join(target_dir, txt_file)
            shutil.move(source_path, dest_path)


if __name__ == '__main__':
    source_dir = 'data/dataset_cleaned'
    data_gen = GenerateDataset(source_dir=source_dir)
    # movendo arquivos... 1325 arquivos para train, 284 arquivos para val e 285 arquivos para test
    


