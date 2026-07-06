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

import gru_class as gru_class
import cnn_class as cnn_class
smoothie = SmoothingFunction().method4
import prepare_data as prepare_data

tokenizer = AutoTokenizer.from_pretrained("t5-base")

vocab_size = tokenizer.vocab_size
hidden_size = 256
num_layers = 1
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225))
])

dataset = prepare_data.Flickr8kDataset(
    captions_file="Flickr8k/captions.txt",
    images_path="Flickr8k/Images",
    tokenizer=tokenizer,
    transform=transform
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train, test = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=False)

# CNN model to extract initial features, RNN to classify correct subtitle
cnn_model = cnn_class.CNNExtractor().to(device)

rnn_model = gru_class.GRU(vocab_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
# Optimize the RNN and CNN
optimizer = torch.optim.Adam(list(rnn_model.parameters()) + list(cnn_model.parameters()),
                              lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    rnn_model.train()
    for i, (image, input, attention_mask) in enumerate(train_loader):
        images = image.to(device)
        input_id = input.to(device)
        mask = attention_mask.to(device)

        # Forward pass
        cnn_output = cnn_model(images)
        hidden = cnn_output.unsqueeze(0)
        
        # Shifting the input ids
        inputs = input_id[:, :-1]  # Remove last token (typically EOS or padding)
        targets = input_id[:, 1:]  # align with shifted

        rnn_output = rnn_model(inputs, hidden)

        # Remove masking tokens when calculating loss
        mask = mask[:, 1:]
        # Backward and optimization
        loss = criterion(rnn_output.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:  
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    rnn_correct, rnn_samples = 0, 0
    rnn_model.eval()

    for images, input, attention_mask in test_loader:
        images = images.to(device)
        input_id = input.to(device)
        attention_mask = attention_mask.to(device)
        
        cnn_output = cnn_model(images)
        generated_ids = rnn_model.generate(cnn_output)  

        target = tokenizer.batch_decode(input_id, skip_special_tokens=True)
        predict = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for pred_cap, tar_cap in zip(predict, target):
            pred_token = pred_cap.lower().split()
            tar_token = tar_cap.lower().split()
            
            # Placed in the right order and without the list the accuracy goes down
            bleu = sentence_bleu([tar_token], pred_token, smoothing_function=smoothie)

            if bleu >= 0.20:
                rnn_correct += 1
            rnn_samples += 1

# Calculate accuracy
rnn_accuracy = 100.0 * (rnn_correct / rnn_samples)
print(f'RNN Model accuracy: {rnn_accuracy}%')

PATH = 'rnn_model.pth'

# Save the model
torch.save(rnn_model.state_dict(), PATH)
