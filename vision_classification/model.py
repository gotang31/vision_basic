import torch
import torchvision
from torch import nn

class MyResNet(torchvision.models.ResNet):
    def __init__(self, block, layers):
        super(MyResNet, self).__init__(block, layers)

    def forward(self, img):
      '''
      img : img tensor, (N, C, H, W)
      box : box coordinate, (x1, y1, x2, y2)
      '''
      x = self.conv1(img)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      x = self.avgpool(x)
      x = torch.flatten(x, 1)
      x = self.fc(x)

      return x

def _resnet(block, layers, weights, progress, num_classes, mode, freeze_depth):
    '''
    block: BasicBlock or Bottleneck
    layers: block layer num order
    weights: pretrained model weights
    progress: False or True (displays a progress bar of the download to stderr)
    mode: whether last fc layer is new or not. True is new, False is not.
    '''

    model = MyResNet(block, layers)

    if weights is not None: # pre-trained weights load
        model.load_state_dict(weights.get_state_dict(progress = progress, check_hash=True))

    if mode == True:
        model.fc = nn.Linear(512 * block.expansion, num_classes)

    else:
        model.fc.out_features = num_classes
    
    if freeze_depth:
        last_child_name = None
        for name, layer in list(model.named_children())[:-(freeze_depth-1)]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
            last_child_name = name
            
        print(f'{last_child_name} is last freezed layer ')
        print(f'----Non-freezed and trainable layer----')
        for name, layer in list(model.named_children())[-(freeze_depth-1):]:
            print(name)

    return model


def SimRes50(num_classes, weights = None, progress = True, mode = False, freeze_depth = None):

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    Bottleneck = torchvision.models.resnet.Bottleneck

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, num_classes, mode, freeze_depth)