import torch
import torch.nn as nn
import torchvision.models as models


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x


class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=4, hidden_size=128, num_layers=1):
        super(CNN_LSTM_Model, self).__init__()

        self.cnn = CNNFeatureExtractor()

        self.lstm = nn.LSTM(
            input_size=self.cnn.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()

        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        final_output = lstm_out[:, -1, :]

        output = self.classifier(final_output)
        return output
