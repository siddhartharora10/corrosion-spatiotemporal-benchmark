import torch
import torch.nn as nn
import torchvision.models as models


class CNN_Only_Model(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_Only_Model, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, C, H, W)
        # Take ONLY last frame (simulate no temporal modeling)
        x = x[:, -1, :, :, :]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
