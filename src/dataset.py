import cv2
import torch
import os
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.ids = []

        for img in os.listdir(self.directory):
            self.ids.append(img.split("_")[0])
        
        ids_set = set(self.ids)
        self.ids = [elt for elt in ids_set]
        self.ids.sort()


    def __len__(self):
        return len(self.ids)
    

    def __getitem__(self, idx):
        id = self.ids[idx]
        img_path = os.path.join(self.directory, id + "_sat.jpg")
        mask_path = os.path.join(self.directory, id + "_mask.png")

        # Lecture image/mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalisation
        img = img.astype('float32')/255.0
        mask = mask.astype('float32')/255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        
        # Changer l'ordre des axes
        img = img.transpose(2, 0, 1)

        # Conversion en Tensor
        img_tensor = torch.tensor(img)
        mask_tensor = torch.tensor(mask)
        mask_tensor =  mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor