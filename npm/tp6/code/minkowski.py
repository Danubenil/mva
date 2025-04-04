
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

import numpy as np
import random
import math
import os
import time
import torch
from tqdm import tqdm
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms#, utils
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../../MinkowskiEngine')
import examples.classification_modelnet40 as mi
# Import functions to read and write ply files
from ply import write_ply, read_ply



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train"):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = torch.tensor(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}



class MLP(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, classes)
        )
    def forward(self, input):
        
        return self.MLP(input)


class PointNetBasic(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()

        self.shared_MLP1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1), 
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.shared_MLP2 = nn.Sequential (
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1024)
        )

        self.global_feat = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, classes)
        )
        

    def forward(self, input):
        out = self.shared_MLP1(input)
        out = self.shared_MLP2(out)
        out = self.global_feat(out)
        return out

        
        
        
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        # YOUR CODE
        self.shared_MLP = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1024)
        )

        self.global_MLP = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, k*k)
        )
        self.k = k

    def forward(self, input):
        out = self.shared_MLP(input)
        out = self.global_MLP(out)
        out = out.reshape((-1, self.k, self.k)) 
        
        return out

class PointNetFull(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        # YOUR CODE
        self.tnet = Tnet(k=3)
        self.basic_pointnet = PointNetBasic(classes=classes)

    def forward(self, input):
        # YOUR CODE 
        
        matrix_transform = self.tnet(input)
        inputs = torch.matmul(input.transpose(1,2), matrix_transform)
        
        out = self.basic_pointnet(inputs.transpose(1, 2))

        return out, matrix_transform

def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.0001):
    criterion = torch.nn.CrossEntropyLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)



def train(model, device, train_loader, test_loader=None, epochs=250, lr = 0.0005, pointnetloss = False, step_size = 20):
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= step_size , gamma=0.5)
    loss=0
    losses_train = []
    losses_test = []
    for epoch in tqdm(range(epochs)): 
        model.train()
        loss_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            
            
            if not pointnetloss:
                outputs = model(inputs.transpose(1,2))
                loss = basic_loss(outputs, labels)
            else:
                outputs, m3x3 = model(inputs.transpose(1,2))
                loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() / train_loader.batch_size
        scheduler.step()
        losses_train.append(loss_total / len(train_loader))
        model.eval()
        correct = total = 0
        test_acc = 0
        loss_total = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    
                    if not pointnetloss:
                        outputs = model(inputs.transpose(1,2))
                        loss = basic_loss(outputs, labels)
                    else:
                        outputs, m3x3 = model(inputs.transpose(1,2))
                        loss = pointnet_full_loss(outputs, labels, m3x3)
                    loss_total += loss.item() / test_loader.batch_size
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total   
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss_total, test_acc))
            losses_test.append(loss_total / len(test_loader))
    return losses_train, losses_test


 
if __name__ == '__main__':
    
    t0 = time.time()
    
    ROOT_DIR = "../data/ModelNet40_PLY"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    device = "cuda"
    print("Device: ", device)
    
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train')
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test')
    print(test_ds[0]['pointcloud'].shape)
    print(test_ds[1]['pointcloud'].shape)
    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
    minkowski_used = True
    if minkowski_used:
        config = {
    "voxel_size": 0.05,
    "max_steps": 100000,
    "val_freq": 1000,
    "batch_size": 32,
    "lr": 1e-1,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "stat_freq": 100,
    "weights": "modelnet.pth",
    "seed": 777,
    "translation": 0.2,
    "test_translation": 0.0,
}
        class MyObject:
            def __init__(self, d=None):
                if d is not None:
                    for key, value in d.items():
                        setattr(self, key, value)
        config = MyObject(config)
        model = mi.MinkowskiFCNN(in_channel=3, out_channel=40, embedding_channel=1024).to(device)
        mi.train(model, device, config)
        acc = mi.accuracy(model, device, config, phase = "Test")
        print("Accuracy with minkowski : ", acc)
    train_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True, pin_memory = True)
    test_loader = DataLoader(dataset=test_ds, batch_size=128, pin_memory = True)


    
    


