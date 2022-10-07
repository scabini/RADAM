#%%
import timm 
import torch
from torch import _nnpack_available
from torch.nn import Module
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from models import *

LAYERS_NO_ACT = {
    'quarter': {
        'resnet18': 'layer1.1.add',
        'resnet50': 'layer1.2.add',
        'resnetv2_50x1_bit_distilled': 'stages.0.blocks.2.add',
        'convnext_large_in22k': '',
        'vgg19': '',
    },
    'middle': {
        'resnet18': 'layer2.1.add',
        'resnet50': 'layer2.3.add',
        'resnetv2_50x1_bit_distilled': 'stages.1.blocks.3.add',
        'convnext_large_in22k': '',
        'vgg19': '',
    },
    'top_quarter': {
        'resnet18': 'layer3.1.add',
        'resnet50': 'layer3.5.add',
        'resnetv2_50x1_bit_distilled': 'stages.2.blocks.5.add',
        'convnext_large_in22k': '',
        'vgg19': '',
    }, 
    'last': {
        'resnet18': 'layer4.1.add',
        'resnet50': 'layer4.2.add',
        'resnetv2_50x1_bit_distilled': 'stages.3.blocks.2.add',
        'convnext_large_in22k': '',
        'vgg19': '',
    }
}


#%%
def get_feature_dict(model, features_depth):
    layers = {}
    for depth in features_depth:
        layers[LAYERS_NO_ACT[depth][model]] = depth
    return layers
        
        

class extractor_pretrained_features(Module):
    def __init__(self, model, features_depth=['last']):
        super(extractor_pretrained_features, self).__init__()
        self.net = timm.create_model(model, pretrained=True, exportable=True)
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.net.eval()
        self.layers = get_feature_dict(model, features_depth)
        self.feature_extractor = create_feature_extractor(self.net, return_nodes=self.layers)
            
        
        
    def forward(self, x):
        return self.feature_extractor(x)
    
    def get_features(self, x, depth='last', pooling='AvgPool2d', Q=1):
        self.net.eval()
        fmap = self.feature_extractor(x)
        fmap = fmap[depth]
        
        if 'ELM' in pooling:
            return torch.flatten(POOLINGS[pooling](Q)(fmap),
                                 start_dim=1, end_dim=-1)
            
        else:
            return torch.flatten(POOLINGS[pooling](fmap.size()[2])(fmap),
                                 start_dim=1, end_dim=-1)
        
    
class extractor_random_features(extractor_pretrained_features):
    def __init__(self, model, features_depth=['last']):
        super(extractor_random_features, self).__init__(model, features_depth=features_depth)
        self.net = timm.create_model(model, pretrained=False, exportable=True)
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.net.eval()
        
class extractor_isotropic_features(Module):
    def __init__(self, model, features_depth=['last'], output_stride=8):
        super(extractor_isotropic_features, self).__init__()
        if output_stride == -1:
            self.net = timm.create_model(model, pretrained=True, exportable=True)
        else:
            try:
                self.net = timm.create_model(model, pretrained=True, exportable=True, output_stride = 8)
            except:
                try:
                    self.net = timm.create_model(model, pretrained=True, exportable=True, output_stride=32)
                except:
                    self.net = timm.create_model(model, pretrained=True, exportable=True)
        for param in self.net.parameters():
            param.requires_grad = False
            
        self.net.eval()
        self.layers = get_feature_dict(model, features_depth)
        self.feature_extractor = create_feature_extractor(self.net, return_nodes=self.layers)
        
    def forward(self, x):
        return self.net(x)
    
    def get_features(self, x, depth='last', pooling='AvgPool2d', Q=1): 
        self.net.eval() #this is needed to disable batch norm, dropout, etc
        fmap = self.feature_extractor(x)
        fmaplist = [v for _,v in fmap.items()]
        
        if 'ELM' in pooling: #only works with ELMfatspectral (aggregative)
            return torch.flatten(POOLINGS[pooling](Q)(fmaplist),
                             start_dim=1, end_dim=-1)
        else: #only works with aggregation GAP
            return torch.flatten(POOLINGS[pooling]()(fmaplist),
                                  start_dim=1, end_dim=-1)

    
    
#%%
# example of use and comparison with previous implementation
# model_extractor = extractor_pretrained_features(model='resnet50', features_depth=['last'])
# model = timm_pretrained_features(model='resnet50')
# x = torch.randn((128,3,224,224))
# x2 = torch.tensor(x)

# output = model.get_features(x, pooling="ELMaggregativeEnsembleM20")
# output_extractor = model_extractor.get_features(x, pooling="ELMaggregativeEnsembleM20")


# print("output",output[-1])
# print("output_extractor",output_extractor)


# %%
