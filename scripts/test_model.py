import torch
from model_cnn_lstm import CNN_LSTM_Model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = CNN_LSTM_Model().to(device)

dummy_input = torch.randn(2, 5, 3, 224, 224).to(device)

output = model(dummy_input)

print("Output shape:", output.shape)
