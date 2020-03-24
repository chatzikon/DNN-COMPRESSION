import sys

dataset='cifar10'


if dataset=='cifar10':
    sys.path.append('../step1/cifar10')
    from data_loader import get_test_loader
elif dataset=='cifar100':
    sys.path.append('../step1/cifar100')
    from data_loader_100 import get_test_loader



sys.path.append('../')


import torchnet as tnt
from torch.autograd import Variable
import torch.nn.functional as F


from model_utils.load_utils import load_model, SAVE_ROOT
from model_utils.model_utils import get_layer_names






MODEL_NAME='resnet56_cifar10'
#MODEL_NAME='resnet56_cifar100'

#MODEL_NAME='resnet110_cifar10'
#MODEL_NAME='resnet110_cifar100'

#MODEL_NAME='mobilenetv1_cifar10'
#MODEL_NAME='mobilenetv1_cifar100'

#MODEL_NAME='mobilenetv2_cifar10'
#MODEL_NAME='mobilenetv2_cifar100'






model_init,model = load_model(MODEL_NAME)


layer_names, conv_layer_mask = get_layer_names(model,'conv')
layer_names_bn, bn_layer_mask = get_layer_names(model,'batchnorm')
fc_layer_mask = (1 - conv_layer_mask).astype(bool)




print(model)







bs = 64



from tensor_compression import get_compressed_model

import copy
import torch
import os
import numpy as np


#no decomposition of first layer and linear layers
def split_resnet_layers_by_blocks(lnames):

    starts = ['layer{}'.format(i) for i in range(1, 4)]
    start_idx = 1
    blocks_idxs = []
    layer_names_by_blocks = []

    for s in starts:
        curr_block =  [l for l in lnames if l.startswith(s)]
        layer_names_by_blocks.append(curr_block)

        blocks_idxs.append(np.arange(start_idx, start_idx+len(curr_block)))
        start_idx += len(curr_block)

    return blocks_idxs





CONV_SPLIT = 3
n_layers = len(layer_names)
n_layers_bn = len(layer_names_bn)




#decomposition_conv = 'cp3'
decomposition_conv = 'tucker2'




WEAKEN_FACTOR = None

#X_FACTOR used:
# res56_c10: 2.32
# res56_c100: 1.587
# res56_c10: 1.71
# res56_c100: 1.395


X_FACTOR = 2.32
rank_selection_suffix = "{}x".format(X_FACTOR)





#specify rank of each layer

if  MODEL_NAME=='resnet56_cifar10' or MODEL_NAME=='resnet110_cifar10' or MODEL_NAME=='resnet56_cifar100' or MODEL_NAME=='resnet110_cifar100':


    ranks_conv = [None if not (name.endswith('conv1') or name.endswith('conv2') ) else -X_FACTOR
                  for name in layer_names[conv_layer_mask]]


elif  MODEL_NAME=='mobilenetv1_cifar10' or MODEL_NAME=='mobilenetv1_cifar100' :


    ranks_conv = [None if not ( name.endswith('conv2') ) else -X_FACTOR
                  for name in layer_names[conv_layer_mask]]

elif MODEL_NAME=='mobilenetv2_cifar10' or MODEL_NAME=='mobilenetv2_cifar100':

    ranks_conv = [None if not (name.endswith('conv1') or name.endswith('conv3') or name.endswith('0')) else -X_FACTOR
                  for name in layer_names[conv_layer_mask]]













ranks = np.array([None] * len(layer_names))
ranks[conv_layer_mask] = ranks_conv



decompositions = np.array([None] * len(layer_names))
decompositions[conv_layer_mask] = decomposition_conv



SPLIT_FACTOR = CONV_SPLIT
save_dir = "{}/models_finetuned/{}/{}/{}/layer_groups:{}".format(SAVE_ROOT,MODEL_NAME,
                                                                                  decomposition_conv,
                                                                                  rank_selection_suffix,
                                                                                  SPLIT_FACTOR)



if not os.path.exists(save_dir):
    os.makedirs(save_dir)


device = 'cuda'



RESNET_SPLIT = True
if (MODEL_NAME =='resnet56_cifar10' or MODEL_NAME =='resnet110_cifar10' or MODEL_NAME =='resnet56_cifar100' or MODEL_NAME =='resnet110_cifar100') and (RESNET_SPLIT):

    split_tuples = split_resnet_layers_by_blocks(layer_names[conv_layer_mask])[:]
    split_tuples_bn = split_resnet_layers_by_blocks(layer_names_bn[bn_layer_mask])[:]

else:
    split_tuples = np.array_split(np.arange(n_layers)[conv_layer_mask], CONV_SPLIT)[::-1]
    split_tuples.reverse()







compressed_model = copy.deepcopy(model)


print(ranks)



for local_iter, tupl in enumerate(split_tuples):


    lname,lname_bn, rank, decomposition = layer_names[tupl], layer_names_bn[tupl],ranks[tupl],  decompositions[tupl]

    if isinstance(tupl[0], np.ndarray):
        print(lname, tupl[0])

    compressed_model = get_compressed_model(MODEL_NAME,compressed_model,
                                   ranks=rank,  layer_names = lname, layer_names_bn = lname_bn,
                                    decompositions = decomposition,
                                    vbmf_weaken_factor = WEAKEN_FACTOR,return_ranks=True)


print(compressed_model)

#
filename = "{}/t.pth.tar".format(save_dir)
torch.save(compressed_model,filename)
print(filename)

def test(model,test_loader):
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    correct = 0
    with torch.no_grad():
        for data, target,index in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss=F.cross_entropy(output, target)
            test_loss.add(loss.item()) # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        loss.item(), correct, len(test_loader.sampler),
        100. * float(correct) / len(test_loader.sampler)))
    return float(correct) / float(len(test_loader.sampler))




if dataset=='cifar10':

    test_loader = get_test_loader('../step1/cifar10/cifar10',
                        128,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

elif dataset=='cifar100':

    test_loader = get_test_loader('../step1/cifar100/cifar100',
                                  128,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)






from collections import defaultdict



def count_params(model):
    n_params = defaultdict()

    for name, param in model.named_parameters():
        n_params[name] = param.numel()
    return n_params


def count_params_by_layers(params_count_dict):
    params_count_dict_modif = defaultdict()



    for k, v in params_count_dict.items():
        if '-' not in k:
            k_head = k.strip('.weight').strip('.bias')
            try:
                params_count_dict_modif[k_head] += params_count_dict[k]
            except:
                params_count_dict_modif[k_head] = params_count_dict[k]
        else:
            k_head = '.'.join(k.split('-')[0].split('.')[:-1])
            try:
                params_count_dict_modif[k_head] += params_count_dict[k]
            except:
                params_count_dict_modif[k_head] = params_count_dict[k]

    return params_count_dict_modif


params_count_dict_m = count_params(model)
params_count_dict_cm = count_params(compressed_model)
params_count_dict_m_init = count_params(model_init)



num_parameters = sum([param.nelement() for param in compressed_model.parameters()])
num_parameters1 = sum([param.nelement() for param in model.parameters()])
num_parameters2 = sum([param.nelement() for param in model_init.parameters()])


print('Params, a:initial, b:pruned, c:decomposed ')

x1=sum(params_count_dict_m.values())/sum(params_count_dict_cm.values())
x11=sum(params_count_dict_m_init.values())/sum(params_count_dict_cm.values())

print('a: '+str(sum(params_count_dict_m_init.values())))
print('a: '+str(num_parameters2))

print('b: '+str(sum(params_count_dict_m.values())))
print('b: '+str(num_parameters1))

print('c: '+str(sum(params_count_dict_cm.values())))
print('c: '+str(num_parameters))


print('Params ratio, a:initial/decomposed, b:pruned/decomposed')

print('a: '+str(x11))
print('b: '+str(x1))

print('a: '+str(num_parameters2/num_parameters))
print('b: '+str(num_parameters1/num_parameters))

print('Params pruned, a:decomposed to initial, b:decomposed to pruned')

print('a: '+str(1-num_parameters/num_parameters2))
print('b: '+str(1-num_parameters/num_parameters1))








#
import sys
sys.path.append("../")

from flopco import FlopCo

model.cpu()
model_init.cpu()
compressed_model.cpu()

if 'cifar10' in MODEL_NAME:
    flopco_m = FlopCo(model.eval(), img_size=(1, 3, 32, 32), device='cpu')
    flopco_m_init = FlopCo(model_init.eval(), img_size=(1, 3, 32, 32), device='cpu')

    flopco_cm = FlopCo(compressed_model.eval(), img_size=(1, 3, 32,32), device='cpu')



elif 'imagenet' in MODEL_NAME:
    flopco_m = FlopCo(model, img_size=(1, 3, 224, 224), device='cpu')
    flopco_m_init = FlopCo(model_init, img_size=(1, 3, 224, 224), device='cpu')


    flopco_cm = FlopCo(compressed_model, img_size=(1, 3, 224, 224), device='cpu')




print('FLOPs a:init/decomposed, b:pruned/decomposed')
print('a: '+str(flopco_m_init.total_flops / flopco_cm.total_flops))
print('b: '+str(flopco_m.total_flops / flopco_cm.total_flops))

print('FLOPs pruned, a:decomposed to initial, b:decomposed to pruned')
print('a: '+str(1-flopco_cm.total_flops/flopco_m_init.total_flops) )
print('b: '+str(1-flopco_cm.total_flops/flopco_m.total_flops) )







