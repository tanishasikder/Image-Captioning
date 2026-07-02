import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Embedding layer to convert token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # GRU to process the sequence of word embeddings
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        #self.gru = nn.GRU(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        #self.fc = nn.Linear(vocab_size, hidden_size)

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