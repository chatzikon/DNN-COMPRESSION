from __future__ import print_function
import argparse
import os
import torch.nn as nn
import torch



import torchvision


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', default='mobilenetv2', type=str,
                    help='architecture to use')
parser.add_argument('--strategy', default='P1', type=str,
                    help='metric to use')

args = parser.parse_args()

model=torchvision.models.mobilenet_v2(pretrained=True)

if args.refine:
    checkpoint = torch.load(args.refine)
    model = mobilenet_v2(inverted_residual_setting=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])



if args.resume:
     if os.path.isfile(args.resume):
         checkpoint = torch.load(args.resume)
         model.load_state_dict(checkpoint['state_dict'])
     else:
         print('no checkpoint')




from scipy import stats

a=[]
T_min=2
T_max=-1

count=0

#find the Tmin and Tmax of section 3.4

for m in model.modules():
    if isinstance(m, (nn.Conv2d)):

        weight_copy = m.weight.data.clone()
        weight_copy = weight_copy.cpu().numpy()
        a= weight_copy.flatten()

        if args.strategy == 'P1':
            W, p = stats.shapiro(a)
        elif args.strategy == 'P2':
            W, p = stats.jarque_bera(a)
        elif args.strategy == 'P3':
            W = stats.kstat(a, 4) * stats.kstat(a, 3)



        if W>T_max:
            T_max=W
        if W<T_min:
            T_min=W


print('T_max')
print(T_max)
print('T_min')
print(T_min)

