import argparse
import numpy as np
import os
import torchnet as tnt

import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy import stats
from models import resnet
from data_loader import get_train_valid_loader, get_test_loader
from compute_flops import print_model_param_flops
import torch.nn.functional as F



# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the resnet')
parser.add_argument('--model', default='./pretrained_models/resnet_model_best_acc_0.9264.pth.tar', type=str, metavar='PATH',
                    help='path to the pretrained model (default: none)')
parser.add_argument('--save', default='./pruned_models', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet(depth=args.depth, dataset=args.dataset)


if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model = resnet(depth=args.depth, dataset=args.dataset)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'  Prec1: {:f}"
              .format(args.model, best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print('Pre-processing Successful!')



if args.cuda:
     model.cuda()






test_loader = get_test_loader('./cifar10',
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






layer_id = 1
cfg = []
cfg_mask = []

total=0

#calculating the total number of network filters
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        if layer_id % 2 == 0:
            total += m.weight.data.shape[0]
        layer_id += 1

stack=torch.zeros(total)


counter=0
layer_id = 1


#stack the cumulant value of each filter
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        if layer_id % 2 == 0:
            weight_copy1 = m.weight.data.clone().cpu().numpy()
            for i in range(weight_copy1.shape[0]):


                stack[counter] = stats.kstat(weight_copy1[i, :, :, :], 4) * stats.kstat(weight_copy1[i, :, :, :], 3)
                counter += 1
        layer_id += 1


y, i = torch.sort(stack)
percent=0.32
thre_index = int(total * percent)
thre = y[thre_index]
print(thre)
pruned=0
layer_id=1

#decide which channels to prune
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]
        # prune only the first convolutional layer of each residual block
        if layer_id % 2 == 0:
            num_keep=0
            weight_copy1 = m.weight.data.clone().cpu().numpy()
            mask=torch.zeros(out_channels)
            for i in range(out_channels):
                temp=stats.kstat(weight_copy1[i, :, :, :], 4) * stats.kstat(weight_copy1[i, :, :, :], 3)
                if temp>thre:
                    mask[i]=1
                    num_keep+=1
            cfg_mask.append(mask)
            cfg.append(num_keep)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
            layer_id += 1
            continue
        layer_id += 1


newmodel = resnet(dataset=args.dataset, depth=args.depth, cfg=cfg)
if args.cuda:
    newmodel.cuda()



#transfer weights from the pretrained network to the pruned one


start_mask = torch.ones(3)
layer_id_in_cfg = 0
conv_count = 1
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):
        if conv_count == 1:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if conv_count % 2 == 0:
            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue
    elif isinstance(m0, nn.BatchNorm2d):
        if conv_count % 2 == 1:
            mask = cfg_mask[layer_id_in_cfg-1]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))

            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            continue
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

print('CFG')
print(cfg)

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'res56_global.pth.tar'))

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
with open(os.path.join(args.save, "res56_global.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(float(acc))+"\n")