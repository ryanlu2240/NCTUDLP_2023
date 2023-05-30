import os
import csv
import numpy as np
from PIL import Image
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# default_transform = transforms.Compose([
# 	transforms.ToTensor(),
# ])

default_transform = transforms.Compose([
    # transforms.RandomResizedCrop((64, 64), scale=(0.8, 1.0)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class dataset(Dataset):
    def __init__(self, args, mode="", transform=default_transform):
        assert mode == "train" or mode == "test" or mode == "validate"
        #raise NotImplementedError
			
    def __len__(self):
        raise NotImplementedError
		
    def get_img(self):
        raise NotImplementedError
	
    def get_label(self):
        raise NotImplementedError
	
    def __getitem__(self, index):
        self.set_seed(index)
        img  = self.get_img()
        label = self.get_label()
        return img, label

class iclevr_dataset(dataset):
    def __init__(self, args, mode="train", transform=default_transform):
        super(iclevr_dataset, self).__init__(args, mode, transform)
		
        self.args = args
        self.mode = mode
        self.transforms = transform

		## Get all data paths
        self.train_img_dirs = []
        self.train_labels = []
        obj_data = json.load(open(args.objects_json))

        for key, value in json.load(open(args.train_data_json)).items():
            self.train_img_dirs.append(args.dataset_root + key)
            onehot = torch.zeros((24))
            for cls in value:
                onehot[obj_data[cls]] = 1
            self.train_labels.append(onehot)


    def __len__(self):
        return len(self.train_labels)

    def get_img(self, index):
        img_path = self.train_img_dirs[index]
        img_arr = Image.open(img_path).convert('RGB')
        img_tensor = self.transforms(img_arr)

        return img_tensor

    def get_label(self, index):
        return self.train_labels[index]
        

    def __getitem__(self, index):
        img  = self.get_img(index)
        label = self.get_label(index)
        return img, label
