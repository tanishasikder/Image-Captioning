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

smoothie = SmoothingFunction().method4

# Load in captions and images separately
images = "Flickr8k/Images"
captions = "captions.txt"
tester = "Flickr8k/Images/667626_18933d713e.jpg"

tester_img = Image.open(tester)
tokenizer = AutoTokenizer.from_pretrained("t5-base")

vocab_size = tokenizer.vocab_size
hidden_size = 256
num_layers = 1
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std=(0.229, 0.224, 0.225))
])

dataset = Flickr8kDataset(
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

# CNN to extract image features
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
    
cnn_model = CNNExtractor().to(device)
#image = tester_img.to(device)
#features = cnn_model(tester_img)

class GRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Embedding layer to convert token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # GRU to process the sequence of word embeddings
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedding = self.embedding(x)
        # THE ERROR HAPPENS AT THE LINE BELOW
        out, hidden = self.gru(embedding, hidden)
        # Pass to a fully connected layer
        out = self.fc(out)
        return out
    
    def init_hidden(self, batch_size):
        # Initial hidden state
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def generate(self, image_features, max=30, sos_token_id=0):
        batch_size = image_features.size(0)
        hidden = image_features.unsqueeze(0)

        input_token = torch.full((batch_size, 1), sos_token_id).to(device)
        generated_ids = []

        for _ in range(max):
            embedding = self.embedding(input_token)
            output, hidden = self.gru(embedding, hidden)
            logits = self.fc(output.squeeze(1))
            next_token = torch.argmax(logits, dim=1, keepdim=True)
            generated_ids.append(next_token)

            input_token = next_token
        
        generated_ids = torch.cat(generated_ids, dim=1)
        return generated_ids

rnn_model = GRU(vocab_size, hidden_size, num_layers).to(device)
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
            pred_token = pred_cap.split()
            tar_token = tar_cap.split()

            bleu = sentence_bleu([tar_token], pred_token, smoothing_function=smoothie)

            if bleu >= 0.20:
                rnn_correct += 1
            rnn_samples += 1

# Calculate accuracy
rnn_accuracy = 100.0 * (rnn_correct / rnn_samples)
print(f'RNN Model accuracy: {rnn_accuracy}%')

