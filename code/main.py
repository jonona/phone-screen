# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:31:33 2020

@author: jonon
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from confusion_plot import plot_confusion_matrix_from_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
import copy
import os.path as osp

np.random.seed(20)

batch_size=20

root=osp.join(osp.dirname(osp.abspath(__file__)), '..', 'data')

composed = T.Compose([T.Resize(256), T.RandomCrop(227, pad_if_needed=True), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

data=torchvision.datasets.ImageFolder(root=root, transform=composed)

lengths = [int(np.ceil(len(data)*0.8)), int(np.floor(len(data)*0.2))]
trainset, testset = random_split(data, lengths)


train_loader=DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(testset, batch_size=batch_size, shuffle=True)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet34(pretrained=True)

#Freezing the weights of all layers except last 2
cntr=0
for child in model.children():
    cntr+=1
    if cntr < 9:
    	#print(child)
    	for param in child.parameters():
    		param.requires_grad = False


#Updating the classifier
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train(epoch):
    model.train()
    correct = 0
    total_loss=0
    
    for i, (image, target) in train_loader:
        image = image.to(device)
        optimizer.zero_grad()
        
        pred = F.softmax(model(image), dim=-1).max(1)[1]
        correct += pred.eq(target).sum().item()
        
        loss = F.cross_entropy(model(image), target)
        
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        if i == (len(train_loader)-1):
            print('[{}/{}] Loss: {:.4f}'.format(i + 1, len(train_loader), total_loss / len(train_loader)))
    
    return correct / (len(train_loader)*batch_size), loss
        


def test(loader,epoch):
    model.eval()

    y_true=torch.zeros([1,1], dtype=torch.long)
    y_pred=torch.zeros([1,1], dtype=torch.long)
    
    for image, target in loader:
        image = image.to(device)
        y_true=torch.cat((y_true, target.unsqueeze(1)))
        with torch.no_grad():
            pred = model(image).max(1)[1]
            y_pred=torch.cat((y_pred, pred.unsqueeze(1)))
        
    cm=confusion_matrix(y_true.squeeze()[1:], y_pred.squeeze()[1:])
    plot_confusion_matrix_from_data(y_true.squeeze()[1:], y_pred.squeeze()[1:], epoch=epoch, columns=['broken', 'ok'])
    return (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])



max_epoch=50
test_acc_plot=np.zeros(max_epoch)
train_acc_plot=np.zeros(max_epoch)
train_loss=np.zeros(max_epoch)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0


for epoch in range(1, max_epoch+1):
    print('Epoch: {:03d}'.format(epoch))
    train_acc, loss = train(epoch)
    train_acc_plot[epoch-1]=train_acc
    train_loss[epoch-1]=loss
    
    test_acc = test(test_loader,epoch)
    test_acc_plot[epoch-1]=test_acc
    
    if test_acc>best_acc:
        best_acc = test_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    
    print('Train: {:.4f}, Test: {:.4f}'.format(train_acc, test_acc))
    exp_lr_scheduler.step()
    
    
#Save best model
model.load_state_dict(best_model_wts)
torch.save(model, osp.join(osp.dirname(osp.abspath(__file__)),'model.pth'))


#Plot learning curves
plt.figure(figsize=[10,10])
plt.plot(np.arange(1,max_epoch+1),train_acc_plot,  color='red', linestyle='dashed', label='train accuracy')
plt.plot(np.arange(1,max_epoch+1), train_loss,  color='blue', label='train loss')
plt.plot(np.arange(1,max_epoch+1),test_acc_plot,  color='green', label='test accuracy')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylim((0,1))
plt.show()

