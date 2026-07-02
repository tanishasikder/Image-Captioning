# CNN to extract image features
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt
from PIL import Image

#hugging face
from transformers import AutoTokenizer
import random

class CNNExtractor(nn.Module):
    def __init__(self, output_size=256):
        super().__init__()
        # ResNet18 to extract image features
        resnet = models.resnet18(pretrained=True)
        # Removes the last layer (it is classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Resnet18 outputs 512, this makes it output 256
        self.fc = nn.Linear(512, output_size)

    def forward(self, x):
        features = self.resnet(x)
        # Return the features reshaped as [batch_size, feature_dimension]
        features = features.view(features.size(0), -1)
        return self.fc(features)