import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(MaskedHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target, mask):
        # pred: [B, 240, 1], target: [B, 240, 1], mask: [B, 240, 1]
        loss = F.huber_loss(pred, target, delta=self.delta, reduction='none')
        masked_loss = loss * mask
        # Average only over unmasked elements
        mask_sum = mask.sum()
        if mask_sum > 0:
            return masked_loss.sum() / mask_sum
        else:
            return masked_loss.sum() * 0.0

class MultiTaskLoss(nn.Module):
    def __init__(self, temp=2.0):
        super(MultiTaskLoss, self).__init__()
        self.temp = temp
        self.register_buffer('task_weights', torch.ones(3))
        self.prev_losses = None
        self.prev_prev_losses = None

    def update_weights(self, current_losses):
        # current_losses: [L_forearm, L_elbow, L_cls]
        if self.prev_losses is None:
            self.prev_losses = current_losses
            return
        if self.prev_prev_losses is None:
            self.prev_prev_losses = self.prev_losses
            self.prev_losses = current_losses
            return
        
        # Calculate rates of change
        w = torch.zeros(3, device=self.task_weights.device)
        for i in range(3):
            w[i] = self.prev_losses[i] / (self.prev_prev_losses[i] + 1e-8)
            
        # DWA Softmax
        exp_w = torch.exp(w / self.temp)
        self.task_weights = 3.0 * exp_w / torch.sum(exp_w)
        
        # Shift history
        self.prev_prev_losses = self.prev_losses
        self.prev_losses = current_losses

    def forward(self, l_forearm, l_elbow, l_cls):
        # Weighted sum
        loss = (self.task_weights[0] * l_forearm + 
                self.task_weights[1] * l_elbow + 
                self.task_weights[2] * l_cls)
        return loss, self.task_weights.detach().cpu().numpy()
