# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:51:02 2020

@author: jonon
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
#import torch.nn as nn
import torch.nn.functional as F
#import torchvision
import torchvision.transforms as T
import os.path as osp
from PIL import Image
import os
import easygui

np.random.seed(20)

#path = easygui.enterbox("Path to folder with test pictures")
path = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'test')

composed = T.Compose([T.Resize(256), T.RandomCrop(227, pad_if_needed=True), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

filenames=os.listdir(path)

model = torch.load(osp.join(osp.dirname(osp.abspath(__file__)), '..', 'code', 'model.pth'))
model.eval()

for file in filenames:
    full=osp.join(path,file)
    img=Image.open(full)
    img_t=composed(img)
    batch = torch.unsqueeze(img_t, 0)
    out=F.softmax(model(batch), dim=-1)[0]
    #out=np.array([0.0,1.0])
    fig=plt.figure()
    fig.suptitle('Broken: {:.4f}\n Unbroken: {:.4f}'.format(out[0],out[1]))
    plt.axis('off')
    img=np.array(img, dtype=float)/255
    plt.imshow(img)
    plt.savefig(osp.join(path, 'results', file))

    