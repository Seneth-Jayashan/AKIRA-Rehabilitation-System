import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        # Causal Conv1
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # We use LayerNorm over the channel dimension (requires transpose)
        self.ln1 = nn.GroupNorm(1, n_outputs) # GroupNorm with 1 group == LayerNorm over channels for Conv1D

        # Causal Conv2
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.GroupNorm(1, n_outputs)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.ln1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.ln2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class MultiTaskTCN(nn.Module):
    def __init__(self, num_inputs=12, num_channels=[32, 32, 32, 32, 32], kernel_size=3, dropout=0.1):
        super(MultiTaskTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
        # Heads
        self.forearm_head = nn.Conv1d(num_channels[-1], 2, kernel_size=1)
        self.elbow_head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)
        self.class_head = nn.Conv1d(num_channels[-1], 20, kernel_size=1)

    def forward(self, x):
        # x is [B, 240, 12] -> [B, L, C]
        # Pytorch Conv1d expects [B, C, L]
        x = x.transpose(1, 2)
        
        # Shared temporal representation [B, 32, 240]
        shared_feat = self.network(x)
        
        # Forearm Output [B, 2, 240] -> transpose to [B, 240, 2]
        forearm_out = self.forearm_head(shared_feat).transpose(1, 2)
        
        # Elbow Output [B, 1, 240] -> transpose to [B, 240, 1]
        elbow_out = self.elbow_head(shared_feat).transpose(1, 2)
        
        # Class Output [B, 20, 240] -> Global Max Pool -> [B, 20]
        class_seq = self.class_head(shared_feat) # [B, 20, 240]
        class_out = F.max_pool1d(class_seq, kernel_size=class_seq.size(2)).squeeze(2) # [B, 20]
        
        return forearm_out, elbow_out, class_out

def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_fp32_mb = total_params * 4 / (1024 ** 2)
    return total_params, trainable_params, size_fp32_mb
