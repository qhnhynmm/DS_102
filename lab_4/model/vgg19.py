import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG19_(nn.Module):
    def __init__(self, config):
        super(VGG19_, self).__init__()
        self.num_classes = config['num_classes']
        self.dropout = config['dropout']
        self.vgg19 = models.vgg19(pretrained=True)

        for param in self.vgg19.parameters():
            param.requires_grad = False

        self.vgg19.classifier = nn.Sequential(
            nn.Linear(self.vgg19.classifier[0].in_features, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),  # Batch Normalization
            nn.Dropout(self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),  # Batch Normalization
            nn.Dropout(self.dropout),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.vgg19(x)
        return x


# Triển khai 
# class VGG19_(nn.Module):
#     def __init__(self, config):
#         super(VGG19_, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, config['num_classes'])
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)

#         return x
class VGG19(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vgg19 = VGG19_(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, imgs, labels=None):
        if labels is not None:
            logits = self.vgg19(imgs)
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            logits = self.vgg19(imgs)
            return logits