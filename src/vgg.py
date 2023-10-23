import torch
import torch.nn as nn
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

class VGG(nn.Module):
    def __init__(self, model, classes, image_size=224):
        super(VGG, self).__init__()
        
        if model == 'VGG11':
            self.features = vgg11_bn(pretrained=True).features
        elif model == 'VGG13':
            self.features = vgg13_bn(pretrained=True).features
        elif model == 'VGG16':
            self.features = vgg16_bn(pretrained=True).features
        elif model == 'VGG19':
            self.features = vgg19_bn(pretrained=True).features
        else:
            raise ValueError('Unsupported VGG model')
        
        with torch.no_grad():
            self.flattened_size = self.features(torch.zeros(1, 3, image_size, image_size)).view(-1).shape[0]
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x