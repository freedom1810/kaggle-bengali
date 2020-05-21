import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
import torch.nn.functional as F

class PretrainedCNN(nn.Module):
    def __init__(self,  freeze = False):
        super(PretrainedCNN, self).__init__()

        self.freeze = freeze
        self.base_model = EfficientNet.from_name('efficientnet-b5')
        
        if self.freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False

        hdim = 2048

        self.lin_g = nn.Linear(hdim, 168)
        self.lin_v = nn.Linear(hdim, 11)
        self.lin_c = nn.Linear(hdim, 7)

    def forward(self, x):
        h = self.base_model.extract_features(x)

        h = F.adaptive_avg_pool2d(h, 1)
        h = h.view(h.size(0), -1)
        # print(h.size())
        h_g = self.lin_g(h)
        h_v = self.lin_v(h)
        h_c = self.lin_c(h)
        return h_g, h_v, h_c

class PretrainedCNN1295(nn.Module):
    def __init__(self):
                
        super(PretrainedCNN1295, self).__init__()

        self.base_model = EfficientNet.from_pretrained('efficientnet-b5')
        hdim = 2048

        self.lin = nn.Linear(hdim, 1292)


    def forward(self, x):
        h = self.base_model.extract_features(x)

        h = F.adaptive_avg_pool2d(h, 1)
        h = h.view(h.size(0), -1)
        # print(h.size())
        h = self.lin(h)
        return h


    def extract_feature(self, x):
        h = self.base_model.extract_features(x)

        h = F.adaptive_avg_pool2d(h, 1)
        h = h.view(h.size(0), -1)

        return h

class PretrainedCNNAfterPretrain1292(nn.Module):
    def __init__(self, 
                freeze = False, 
                PATH  =''):
        super(PretrainedCNNAfterPretrain1292, self).__init__()

        self.base_model = PretrainedCNN1295()

        self.PATH = PATH
        if self.PATH != '':
           self.base_model.load_state_dict(torch.load(PATH))

        del self.base_model.lin

        self.freeze = freeze

        if self.freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        hdim = 2048

        self.lin_g = nn.Linear(hdim, 168)
        self.lin_v = nn.Linear(hdim, 11)
        self.lin_c = nn.Linear(hdim, 7)

    def forward(self, x):
        h = self.base_model.extract_feature(x)

        h_g = self.lin_g(h)
        h_v = self.lin_v(h)
        h_c = self.lin_c(h)
        return h_g, h_v, h_c