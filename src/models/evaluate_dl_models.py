import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Conv1DNet(nn.Module):
    def __init__(self, num_classes, in_channels=6):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 50, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LSTMNet(nn.Module):
    def __init__(self, num_classes, input_size=6, hidden_size=128, num_layers=2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class CNN_LSTMNet(nn.Module):
    def __init__(self, num_classes, in_channels=6, hidden_size=128):
        super(CNN_LSTMNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        
        self.lstm = nn.LSTM(128, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x is (B, 6, 200)
        x = self.pool(self.relu(self.conv1(x))) # (B, 64, 100)
        x = self.pool(self.relu(self.conv2(x))) # (B, 128, 50)
        x = x.permute(0, 2, 1) # (B, 50, 128)
        
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class TransformerNet(nn.Module):
    def __init__(self, num_classes, input_size=6, d_model=32, nhead=2, num_layers=1):
        super(TransformerNet, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x is (B, 200, 6)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

def load_data(data_dir="datasets/ml_ready"):
    X_train = np.load(os.path.join(data_dir, "raw_train.npy"))
    X_test = np.load(os.path.join(data_dir, "raw_test.npy"))
    
    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test_df = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    
    classes = sorted(list(set(y_train_df.unique()).union(set(y_test_df.unique()))))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_train = np.array([label_to_idx[l] for l in y_train_df])
    y_test = np.array([label_to_idx[l] for l in y_test_df])
    
    return X_train, y_train, X_test, y_test, classes

def train_and_evaluate(model, model_name, train_loader, X_test, y_test, classes, epochs=15, is_cnn=False):
    print(f"\n{'='*40}")
    print(f"Evaluating Deep Model: {model_name}")
    print(f"{'='*40}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    
    train_time = time.time() - start_time
    
    model.eval()
    start_time = time.time()
    
    if is_cnn:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).transpose(1, 2).to(device)
    else:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, preds = torch.max(outputs, 1)
        
    y_pred = preds.cpu().numpy()
    inference_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    print(f"Training Time  : {train_time:.2f} seconds")
    print(f"Inference Time : {inference_time:.4f} seconds (Total)")
    print(f"Accuracy       : {acc*100:.2f}%")
    print(f"Macro F1-Score : {f1*100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.close()
    
    summary_file = "results/results_summary.csv"
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, 'a') as f:
        if not file_exists:
            f.write("Model,Accuracy,Macro_F1,Train_Time,Inference_Time\n")
        f.write(f"{model_name},{acc:.4f},{f1:.4f},{train_time:.4f},{inference_time:.4f}\n")
        
    return model, acc, f1

def run_track_b():
    print("Loading 3D Tensors for Track B...")
    X_train, y_train, X_test, y_test, classes = load_data()
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    num_classes = len(classes)
    
    X_train_cnn = np.transpose(X_train, (0, 2, 1))
    
    train_dataset_cnn = TensorDataset(torch.tensor(X_train_cnn, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
    
    train_dataset_lstm = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=64, shuffle=True)
    
    cnn_model = Conv1DNet(num_classes=num_classes)
    train_and_evaluate(cnn_model, "1D CNN", train_loader_cnn, X_test, y_test, classes, is_cnn=True)
    
    lstm_model = LSTMNet(num_classes=num_classes)
    train_and_evaluate(lstm_model, "LSTM", train_loader_lstm, X_test, y_test, classes, is_cnn=False)
    
    cnn_lstm_model = CNN_LSTMNet(num_classes=num_classes)
    train_and_evaluate(cnn_lstm_model, "CNN_LSTM", train_loader_cnn, X_test, y_test, classes, is_cnn=True)
    transformer_model = TransformerNet(num_classes=num_classes)
    train_and_evaluate(transformer_model, "Transformer", train_loader_lstm, X_test, y_test, classes, is_cnn=False)

if __name__ == "__main__":
    run_track_b()
