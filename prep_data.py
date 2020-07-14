################################################################################################################################
# ebharucha: 14/7/2020
################################################################################################################################

# Import dependencies
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Define class to create Train & Test data
class TrainTest():
    def __init__(self, img_dir):
        self.classes = []
        imgpath = []
        label = []
        dirs = [x[0] for x in os.walk(img_dir)]
        for d in dirs[1:]:
            class_ = d.split('/')[-1]
            self.classes.append(class_)
            for f in os.listdir(d):
                imgpath.append(os.path.join(f'{d}/{f}'))
                label.append(self.classes.index(class_))
        self.df = pd.DataFrame()
        self.df['imgpath'] = imgpath
        self.df['label'] = label
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_train_test_data(self.df)
    
    def create_train_test_data(self, df):
        X = df.imgpath
        y = df.label
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        return(X_train, X_test, y_train, y_test)

# Define dataset class
class DatasetImg(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        X_arr = Image.open(self.X.iloc[idx])
        y = self.y.iloc[idx]
        if (self.transform):
            X_arr = self.transform(X_arr)
        return (X_arr, y)
    
    def __len__(self):
        return(self.X.shape[0])

# Define class to display sample images from dataloader
# Courtesy Python Engineer https://www.youtube.com/channel/UCbXgNpp0jedKWcQiULLbDTA
class imdisplay():
    def __init__(self, dataloader):
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        self.imshow(torchvision.utils.make_grid(images))

    def imshow(self, img):
        plt.figure(figsize=(15,7))
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

# Initialize parameters
img_dir = './images/'
img_size = 224
batch_size = 10
img_transform = transforms.Compose([transforms.Resize(img_size), 
                                    transforms.RandomCrop(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])                                    

# Create Train, Test split
traintest = TrainTest(img_dir)

# Create Train, Test datasets & dataloaders
dataset_train = DatasetImg(traintest.X_train, traintest.y_train, img_transform)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size)
dataset_test = DatasetImg(traintest.X_test, traintest.y_test, img_transform)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size)
