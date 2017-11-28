import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision







class DeepMask(nn.Module):
    ''' An implementation of the DeepMask segmentation architecture described in arXiv:1506.06204 [cs.CV] '''
    def __init__(self):
        super(DeepMask, self).__init__()
        self.features = partial_vgg_layers()
        
        

        # Segmentation branch weights
        self.segmentation = nn.Sequential(nn.Conv2d(512,512,kernel_size=1), nn.ReLU(inplace=True))
        self.linear_seg_1 = nn.Linear(512*14*14,512)
        self.linear_seg_2 = nn.Linear(512, 56*56)

        # Scoring branch weights
        self.maxpool_scr = nn.MaxPool2d(kernel_size = 2)
        self.score = nn.Sequential(
            nn.Linear(512*7*7,512), 
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024), 
            nn.Dropout(), 
            nn.ReLU(inplace=True), 
            nn.Linear(1024,1)
            )
        
            
    def forward(x):
        x = self.features(x)
        
        # Segmentation branch
        x_segmented = self.segmentation(x)
        x_segmented = x_segmented.view(-1)
        x_segmented = self.linear_seg_1(x_segmented)
        x_segmented = self.linear_seg_2(x_segmented)

        # Scoring branch
        x_score = self.maxpool_scr(x)
        x_score = x_score.view(-1)
        x_score = self.score(x_score)

        return x_segmented, x_score



def partial_vgg_layers():
    '''
        (Copied and modified from torchvision.models.vgg)
        Generates layers for the VGG-A (arXiv:1409.1556 [cs.CV]) architecture 
        used in the DeepMask implementation.
    '''
    layers = []
    in_channels = 3
    for v in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

model = DeepMask()

def pretrained_DeepMask():
    '''
        Loads the pretrained vgg weights from an online model into the necessary DeepMask layers.
        The model should be trained beyond this.
    '''
    print('Loading model...')
    pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg11-bbd30ac9.pth')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Loaded model.')
    return model





