import argparse
import numpy as np
import os
import torchnet as tnt

import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy import stats
from data_loader_100 import get_train_valid_loader, get_test_loader
import torch.nn.functional as F


import sys
sys.path.insert(0, "../cifar10/")
import models
from compute_flops import print_model_param_flops



# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', default='./pretrained_models/mobilenet_c100_0.7631.pth.tar', type=str, metavar='PATH',
                    help='path to the pretrained model (default: none)')
parser.add_argument('--save', default='./pruned_models', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--metric', default='k34', type=str,
                    help='metric to use')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model=models.mobilenet.MobileNet(num_classes=100)


if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model = models.mobilenet.MobileNet(num_classes=100)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'  Prec1: {:f}"
              .format(args.model, best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print('Pre-processing Successful!')



if args.cuda:
     model.cuda()







test_loader = get_test_loader('./cifar100',
                    args.batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True)





# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model,test_loader):
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    correct = 0
    with torch.no_grad():
        for data, target,index in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output= model(data)
            loss = F.cross_entropy(output, target)
            test_loss.add(loss.item())  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss.item(), correct, len(test_loader.sampler),
            100. * float(correct) / len(test_loader.sampler)))
    return float(correct) / float(len(test_loader.sampler)), loss.item()


acc,_= test(model,test_loader)



#decide how many and which filters to prune at each layer


layer_id = 1
cfg = []
cfg_mask = []
coef=4.09



for m in model.modules():

    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        in_channels=m.weight.data.shape[1]
        if (m.kernel_size == (1, 1) and in_channels==1024) :
            cfg_mask.append(torch.ones(out_channels))
            cfg.append(out_channels)
            layer_id += 1
            continue
        elif in_channels==3:
            cfg_mask.append(torch.ones(out_channels))
            cfg.append(out_channels)
            layer_id += 1
            num_keep_pre=32
            continue



        weight_copy = m.weight.data.clone().cpu().numpy()
        a = weight_copy.flatten()
        W, p = stats.shapiro(a)
        prune_prob_stage = np.ceil(10 * (W - 0.25) / (coef * 0.75)) / 10

        if args.metric == 'l1norm':

            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            filter_values = np.sum(weight_copy, axis=(1, 2, 3))


        elif args.metric == 'random':
            weight_copy = m.weight.data.clone().cpu().numpy()




        elif args.metric == 'k3':
            weight_copy = m.weight.data.clone().cpu().numpy()
            filter_values = np.zeros(weight_copy.shape[0])
            for i in range(weight_copy.shape[0]):
                filter_values[i] = stats.kstat(weight_copy[i, :, :, :], 3)


        elif args.metric == 'k4':
            weight_copy = m.weight.data.clone().cpu().numpy()
            filter_values = np.zeros(weight_copy.shape[0])
            for i in range(weight_copy.shape[0]):
                filter_values[i] = stats.kstat(weight_copy[i, :, :, :], 4)


        elif args.metric == 'k34':
            weight_copy = m.weight.data.clone().cpu().numpy()
            filter_values = np.zeros(weight_copy.shape[0])
            for i in range(weight_copy.shape[0]):
                filter_values[i] = stats.kstat(weight_copy[i, :, :, :], 3) * stats.kstat(weight_copy[i, :, :, :], 4)



        elif args.metric == 'skew':
            weight_copy = m.weight.data.clone().cpu().numpy()
            filter_values = np.zeros(weight_copy.shape[0])
            for i in range(weight_copy.shape[0]):
                temp = weight_copy[i, :, :, :].flatten()
                filter_values[i] = stats.skew(temp)


        elif args.metric == 'kur':
            weight_copy = m.weight.data.clone().cpu().numpy()
            filter_values = np.zeros(weight_copy.shape[0])
            for i in range(weight_copy.shape[0]):
                temp = weight_copy[i, :, :, :].flatten()
                filter_values[i] = stats.kurtosis(temp)


        elif args.metric == 'skew_kur':
            weight_copy = m.weight.data.clone().cpu().numpy()
            filter_values = np.zeros(weight_copy.shape[0])
            for i in range(weight_copy.shape[0]):
                temp = weight_copy[i, :, :, :].flatten()
                filter_values[i] = stats.skew(temp) * stats.kurtosis(temp)

        if args.metric == 'random':
            arg_max = list(range(weight_copy.shape[0]))
            np.random.shuffle(arg_max)
        else:
            arg_max = np.argsort(filter_values)




        num_keep = int(out_channels * (1 - prune_prob_stage))



        #no prune of depthwise layers
        if m.kernel_size == (3, 3) and layer_id>2:
            arg_max=arg_max_pre

        if m.kernel_size == (3, 3) :
            num_keep=num_keep_pre




        arg_max_rev = arg_max[::-1][:num_keep]
        mask = torch.zeros(out_channels)
        mask[arg_max_rev.tolist()] = 1
        cfg_mask.append(mask)
        cfg.append(num_keep)
        num_keep_pre=num_keep
        arg_max_pre=arg_max
        layer_id += 1



newmodel = models.mobilenet.MobileNet(num_classes=100,cfg=cfg)

#transfer weights from the pretrained network to the pruned one


start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1
for [m0, m1] in zip(model.modules(), newmodel.modules()):

    if isinstance(m0, nn.Conv2d):
        in_channels = m0.weight.data.shape[1]
        if conv_count == 1 :
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            layer_id_in_cfg += 1
            continue
        if  m0.kernel_size == (1, 1):
            mask_out = cfg_mask[layer_id_in_cfg]
            mask_in = cfg_mask[layer_id_in_cfg - 1]
            idx1 = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
            idx2 = np.squeeze(np.argwhere(np.asarray(mask_in.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            if idx2.size == 1:
                idx2 = np.resize(idx2, (1,))
            w = m0.weight.data[:,idx2.tolist(), :, :].clone()
            w1=w[idx1.tolist(),:,:, :].clone()
            m1.weight.data = w1.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue
        if m0.kernel_size == (3, 3):
            mask_out = cfg_mask[layer_id_in_cfg]
            idx1 = np.squeeze(np.argwhere(np.asarray(mask_out.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            w = m0.weight.data[ idx1.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue

    elif isinstance(m0, nn.BatchNorm2d):
        mask = cfg_mask[layer_id_in_cfg-1]
        idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if idx.size == 1:
            idx = np.resize(idx, (1,))
        m1.weight.data = m0.weight.data[idx.tolist()].clone()
        m1.bias.data = m0.bias.data[idx.tolist()].clone()
        m1.running_mean = m0.running_mean[idx.tolist()].clone()
        m1.running_var = m0.running_var[idx.tolist()].clone()

    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

print('CFG')
print(cfg)
torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'mobilenet_c100.pth.tar'))


num_parameters = sum([param.nelement() for param in newmodel.parameters()])
num_parameters1 = sum([param.nelement() for param in model.parameters()])



flops_std = print_model_param_flops(model, 32)
flops_std1 = print_model_param_flops(newmodel, 32)

print('flops pruned')
print(1-flops_std1/flops_std)
print('params pruned')
print(1-num_parameters/num_parameters1)

acc,_ =test(newmodel,test_loader)

print("number of parameters: "+str(num_parameters))
with open(os.path.join(args.save, "mobilenet_c100.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(float(acc))+"\n")