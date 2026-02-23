import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

from dataset_loader import SequenceDataset
from model_cnn_only import CNN_Only_Model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 8

# Load test dataset
test_dataset = SequenceDataset("../data/sequences/test", augment=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load trained CNN-only model
model = CNN_Only_Model().to(device)
model.load_state_dict(torch.load("../models/cnn_only_model.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

print("\n=== CNN-Only Test Results ===")
print(classification_report(all_labels, all_preds))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

roc = roc_auc_score(np.eye(4)[all_labels], all_probs, multi_class='ovr')
print("ROC-AUC:", roc)
