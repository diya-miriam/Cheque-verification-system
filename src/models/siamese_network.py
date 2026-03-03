import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self,embedding_size=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3),      
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        # Corrected FC
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        return emb1, emb2