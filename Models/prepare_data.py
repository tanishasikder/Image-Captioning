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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

tokenizer = AutoTokenizer.from_pretrained("t5-base")

class Flickr8kDataset(Dataset):
    def __init__(self, captions_file, images_path, tokenizer, transform=None):
        self.caption_dict = {}
        self.images_dir = images_path
        self.transform = transform
        self.tokenizer = tokenizer

        # Open the file name and read it
        with open(captions_file, "r") as f:
            next(f) # Skip the first line
            for caption in f:
                # Split after the first comma
                line = caption.strip().split(',', 1)
                if line[0] not in self.caption_dict:
                    self.caption_dict[line[0]] = []
                self.caption_dict[line[0]].append(line[1])
            
            # List of the file names from caption_dict
            self.image_names = list(self.caption_dict.keys())

    #needs to know total items so it knows how many batches to create
    def __len__(self):
        return len(self.image_names)
    
    #pytorch will look for __getitem__() when it tries to fetch data
    def __getitem__(self, index):
        # Getting the specific image name and captions
        image_name = self.image_names[index]
        caption = random.choice(self.caption_dict[image_name])

        # Creating the file path
        img_path = os.path.join(self.images_dir, image_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Caption is a list of lists. Need one caption
        encoded_caption = self.tokenizer(caption, padding="max_length",
                                         truncation=True, max_length=30,
                                         return_tensors="pt")
        input = encoded_caption["input_ids"].squeeze(0)
        attention_mask = encoded_caption["attention_mask"].squeeze(0)
            
        return image, input, attention_mask