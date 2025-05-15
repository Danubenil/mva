
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
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys

# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud 

class RandomTranslation(object) : 
    def __call__(self, pointcloud):
        t = np.random.uniform(-0.2, 0.2, (1, 3))
        return pointcloud + t

        
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(), ToTensor()])

def test_transforms():
    return transforms.Compose([ToTensor()])



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
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
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
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



def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    for epoch in range(epochs): 
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.transpose(1,2))
            #outputs, m3x3 = model(inputs.transpose(1,2))
            loss = basic_loss(outputs, labels)
            #loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs = model(inputs.transpose(1,2))
                    #outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))


 
if __name__ == '__main__':
    
    t0 = time.time()
    
    ROOT_DIR = "../data/ModelNet10_PLY"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=default_transforms())
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    #model = MLP()
    model = PointNetBasic()
    #model = PointNetFull()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device)
    
    train(model, device, train_loader, test_loader, epochs = 10)
    
    t1 = time.time()
    print("Total time for training : ", t1-t0)

    
    


