import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random


class T2IDataset(Dataset):
    def __init__(self,images,tags):
        self.images = images
        self.tags = tags

    def __len__(self):
        return len(self.images)


    def __getitem__(self,index):
        right_image = self.images[index]
        right_embed = self.tags[index].astype(np.float32)
        wrong_embed = self.find_wrong_embed(right_embed)
        
        return {"right_images" : right_image,
                "right_embed" : right_embed, 
                "wrong_embed" : wrong_embed}
        
    def find_wrong_embed(self, tags):
        while True:
            wrong = np.zeros(15)
            wrong[random.randint(0,5)] = 1
            wrong[random.randint(6,9)] = 1
            wrong[random.randint(10,12)] = 1
            wrong[random.randint(13,14)] = 1
            
            if (tags != wrong).any():
                return wrong.astype(np.float32)
