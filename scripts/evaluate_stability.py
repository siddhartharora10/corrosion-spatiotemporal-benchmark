import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset_loader import SequenceDataset
from model_cnn_lstm import CNN_LSTM_Model
from model_cnn_only import CNN_Only_Model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 8

# Load dataset
test_dataset = SequenceDataset("../data/sequences/test", augment=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load models
hybrid = CNN_LSTM_Model().to(device)
hybrid.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
hybrid.eval()

cnn_only = CNN_Only_Model().to(device)
cnn_only.load_state_dict(torch.load("../models/cnn_only_model.pth", map_location=device))
cnn_only.eval()

def corrupt_sequence(inputs):
    inputs = inputs.clone()

    # Shuffle frames
    if random.random() > 0.5:
        idx = torch.randperm(inputs.size(1))
        inputs = inputs[:, idx, :, :, :]

    # Random frame dropout
    if random.random() > 0.5:
        drop_idx = random.randint(0, inputs.size(1)-1)
        inputs[:, drop_idx] = 0

    return inputs

hybrid_preds = []
cnn_preds = []
labels_all = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        corrupted = corrupt_sequence(inputs)

        out_hybrid = hybrid(corrupted)
        out_cnn = cnn_only(corrupted)

        _, pred_h = torch.max(out_hybrid, 1)
        _, pred_c = torch.max(out_cnn, 1)

        hybrid_preds.extend(pred_h.cpu().numpy())
        cnn_preds.extend(pred_c.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

hybrid_acc = accuracy_score(labels_all, hybrid_preds)
cnn_acc = accuracy_score(labels_all, cnn_preds)

print("\n=== TEMPORAL STABILITY TEST ===")
print("Hybrid Accuracy under jitter:", hybrid_acc)
print("CNN-only Accuracy under jitter:", cnn_acc)
print("Accuracy Drop (Hybrid vs Clean):", 0.96 - hybrid_acc)
print("Accuracy Drop (CNN-only vs Clean):", 0.89 - cnn_acc)
