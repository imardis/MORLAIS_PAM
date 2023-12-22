# Developer Information:
"""
## Developer Profile:
- Name: Vahid Seydi
- GitHub: https://github.com/vahidseydi


## Model Information:
- Model Name: RNN Classifier
- Description: A simplified version of an RNN classifier for identifying marine species based on acoustic signals. 
  The hyperparameters below have been obtained through a complex optimization process.
- Input Size: 4
- Hidden Layer Sizes: h1 = 33, h2 = 27
- Output Size: 2 (for binary classification)
"""

# Importing necessary libraries
import torch.nn as nn
import torch

class RNNClassifier(nn.Module):
    def __init__(self, device):
        super(RNNClassifier, self).__init__()
        self.device = device
        self.input_size = 4
        self.h1 = 33
        self.h2 = 27       

        # Define GRU layers
        self.rnn1 = nn.GRU(self.input_size, self.h1, 1, batch_first=True)
        self.rnn2 = nn.GRU(self.h1, self.h2, 1, batch_first=True)       

        # Fully connected layer
        self.fc = nn.Linear(self.h2, 2)

    def forward(self, x):
        # Initialize hidden states
        h1 = torch.zeros(1, x.size(0), self.h1).to(self.device)
        h2 = torch.zeros(1, x.size(0), self.h2).to(self.device)

        # Pass input through the first GRU layer
        out, _ = self.rnn1(x, h1)
        
        # Pass the output through the second GRU layer
        out, _ = self.rnn2(out, h2)        

        # Extract the last time step output
        out = out[:, -1, :]

        # Pass through the fully connected layer
        out = self.fc(out)

        return out
