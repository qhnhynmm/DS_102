import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50_(nn.Module):
    def __init__(self, config):
        super(ResNet50_, self).__init__()
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        
        self.resnet50 = models.resnet50(pretrained=True)
        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.resnet50.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.resnet50(x)
        return x
# # Define the Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * 4)

#         # Shortcut connection
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels * 4:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * 4)
#             )

#     def forward(self, x):
#         residual = x

#         x = F.relu(self.bn1(self.conv1(x)), inplace=True)
#         x = F.relu(self.bn2(self.conv2(x)), inplace=True)
#         x = self.bn3(self.conv3(x))

#         x += self.shortcut(residual)
#         x = F.relu(x, inplace=True)

#         return x

# # Define the ResNet-50 model
# class ResNet50(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(ResNet50, self).__init__()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(64, 3, stride=1)
#         self.layer2 = self._make_layer(128, 4, stride=2)
#         self.layer3 = self._make_layer(256, 6, stride=2)
#         self.layer4 = self._make_layer(512, 3, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * 4, num_classes)

#     def _make_layer(self, out_channels, blocks, stride):
#         layers = [ResidualBlock(512, out_channels, stride=stride)]
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock(out_channels * 4, out_channels))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x
class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ResNet = ResNet50_(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, imgs, labels=None):
        if labels is not None:
            logits = self.ResNet(imgs)
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            logits = self.ResNet(imgs)
            return logits