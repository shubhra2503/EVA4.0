import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from base import BaseModel
from pdb import set_trace as bp

class QuizDNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dropout_value = 0.1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, padding_mode = 'replicate', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        ) 

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1, padding_mode = 'replicate', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        ) 

        self.pool1 = nn.MaxPool2d(2, 2) 
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1,  padding_mode = 'replicate', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        ) 

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1,  padding_mode = 'replicate', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        ) 
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1,  padding_mode = 'replicate', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        ) 

        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1,  padding_mode = 'replicate', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        ) 

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1,  padding_mode = 'replicate', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        ) 
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1,  padding_mode = 'replicate', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        )         

        self.GAP = nn.AdaptiveAvgPool2d(4)
        self.FC = nn.Linear(3*4*4, 10)
        self.relu = nn.ReLU()

    def forward(self, x1):
        
        x2 = self.convblock1(x1)
        x3 = self.convblock2(x1 + x2)
        x4 = self.pool1(x1 + x2 + x3)
        x5 = self.convblock3(x4)
        x6 = self.convblock4(x4 + x5)
        x7 = self.convblock5(x4 + x5 + x6)
        x8 = self.pool2(x5 + x6 + x7)
        x9 = self.convblock6(x8)
        x10 = self.convblock7(x8 + x9)
        x11 = self.convblock8(x8 + x9 + x10)
        x12 = self.relu(self.GAP(x11))
        x12 = x12.view(x12.size(0), -1)
        x13 = self.FC(x12)
        return x13


