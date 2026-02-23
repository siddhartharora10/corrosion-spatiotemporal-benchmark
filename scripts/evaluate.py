import torch
from torch.utils.data import DataLoader
from dataset_loader import SequenceDataset
from model_cnn_lstm import CNN_LSTM_Model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

test_dataset = SequenceDataset("../data/sequences/test")
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load("../models/best_model.pth"))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

try:
    roc_auc = roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr')
    print("ROC-AUC:", roc_auc)
except:
    print("ROC-AUC calculation skipped")
