import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBaseline(nn.Module):
    def __init__(self, num_inputs=12):
        super(LinearBaseline, self).__init__()
        # Independent per timestep, mapping 12 -> outputs
        self.forearm_head = nn.Linear(num_inputs, 2)
        self.elbow_head = nn.Linear(num_inputs, 1)
        # Class prediction aggregates over time via mean pooling
        self.class_head = nn.Linear(num_inputs, 20)
        
    def forward(self, x):
        # x is [B, 240, 12]
        forearm_out = self.forearm_head(x) # [B, 240, 2]
        elbow_out = self.elbow_head(x)     # [B, 240, 1]
        
        # Mean pooling over time for classification
        x_pool = torch.mean(x, dim=1)      # [B, 12]
        class_out = self.class_head(x_pool)# [B, 20]
        
        return forearm_out, elbow_out, class_out

class LSTMBaseline(nn.Module):
    def __init__(self, num_inputs=12, hidden_size=32):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size, 
                            num_layers=1, batch_first=True)
        
        self.forearm_head = nn.Linear(hidden_size, 2)
        self.elbow_head = nn.Linear(hidden_size, 1)
        self.class_head = nn.Linear(hidden_size, 20)
        
    def forward(self, x):
        # x is [B, 240, 12]
        lstm_out, _ = self.lstm(x) # [B, 240, 32]
        
        forearm_out = self.forearm_head(lstm_out) # [B, 240, 2]
        elbow_out = self.elbow_head(lstm_out)     # [B, 240, 1]
        
        # Classification using the final hidden state or max pooling
        # Using global max pooling for fair comparison with TCN
        lstm_out_t = lstm_out.transpose(1, 2) # [B, 32, 240]
        class_seq = self.class_head(lstm_out) # [B, 240, 20]
        class_out = F.max_pool1d(class_seq.transpose(1, 2), kernel_size=240).squeeze(2) # [B, 20]
        
        return forearm_out, elbow_out, class_out
