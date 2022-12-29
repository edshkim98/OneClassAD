import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet
import config as c
from freia_funcs import *
import time
import timm

WEIGHT_DIR = './weights'
MODEL_DIR = './models/tmp'


def get_cs_flow_model(input_dim=c.n_feat):
    print("Internal feat: ",c.fc_internal)
    nodes = list()
    nodes.append(InputNode(input_dim[0], c.map_size[0][0], c.map_size[0][1], name='input')) #InpuNode -> channel, h, w
    nodes.append(InputNode(input_dim[1], c.map_size[1][0], c.map_size[1][1], name='input2'))
    nodes.append(InputNode(input_dim[2], c.map_size[2][0], c.map_size[2][1], name='input3'))

    for k in range(c.n_coupling_blocks):
        if k == 0:
            node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
        else:
            node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

        nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0, nodes[-1].out1, nodes[-1].out2], parallel_glow_coupling_layer,
                          {'clamp': c.clamp, 'F_class': CrossConvolutions,
                           'F_args': {'channels_hidden': c.fc_internal,
                                      'kernel_size': c.kernel_sizes[k], 'block_no': k}},
                          name=F'fc1_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
    nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
    nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
    nf = ReversibleGraphNet(nodes, n_jac=3)
    return nf

def nf_forward(model, inputs):
    return model(inputs), model.jacobian(run_forward=False)

#@torch.no_grad()
class FeatureExtractor(nn.Module):
    def __init__(self, c):
        super(FeatureExtractor, self).__init__()
        
        if 'se' in c.extractor:
            print("SEResNet!")
            self.feature_extractor = timm.create_model(
                'seresnext50_32x4d',
                pretrained=True,
                features_only=True,
                out_indices=[1,2,3],
            )
        else:
            self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        print("This!!")

    def eff_ext(self, x, use_layer):
        lst = []
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx in use_layer:
                lst.append(x)
            if len(lst) == 3:
                return lst

    def forward(self, x):
        layers =[12,26,35]
        if 'se' in c.extractor:
            out = self.feature_extractor(x)
        else:
            out = self.eff_ext(x,layers)
        
        return out
    
class FeatureExtractor2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.layers = ['_blocks.12','_blocks.26', '_blocks.35'] #12 26 35
        self.feats = []
        for layer_id in self.layers:
            print(layer_id)
            layer = dict([*self.feature_extractor.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self.feats.append(output)
        return fn
    
    def forward(self, x):
        start= time.time()
        _ = self.feature_extractor(x)
        self.feats2 = self.feats
        self.feats = []
        print(time.time()-start)
        return self.feats2
    
class FeatureExtractor3(nn.Module):
    def __init__(self, c):
        super(FeatureExtractor3, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')

    def eff_ext(self, x, use_layer):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == use_layer:
                return x

    def forward(self, x):
        #start= time.time()
        y = list()
        layers =[12,26,35]
        for i,s in enumerate(range(c.n_scales)):
            feat_s = x#F.interpolate(x, size=(c.img_size[0] // (2 ** s), c.img_size[1] // (2 ** s))) if s > 0 else x
            feat_s = self.eff_ext(feat_s,layers[i])
            
            y.append(feat_s)
        #print(len(y), y[0].shape)
#         for i in range(len(y)):
#             print(y[i].shape)
        #print(time.time()-start)
        return y


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model
