import torch.nn as nn
import numpy as np

def get_layer_names(model,type):
    layer_names = []
    conv_layer_mask = []

    if type=='conv':

        for mname, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_names.append(mname)
                conv_layer_mask.append(True)
            elif isinstance(m, nn.Linear):
                layer_names.append(mname)
                conv_layer_mask.append(False)
    elif type=='batchnorm':
        for mname, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d,nn.BatchNorm1d)) :
                layer_names.append(mname)
                conv_layer_mask.append(True)


        
    return np.array(layer_names), np.array(conv_layer_mask)
    
    



