import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms, models
from scipy import stats


import torchvision
from mobilenetv2_imagenet import mobilenet_v2
from compute_flops import count_model_param_flops


# Prune settings
parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save', default='./', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
parser.add_argument('--data', type=str, default='/home/chatziko/imagenet',
                    help='Path to imagenet validation data')
parser.add_argument('--metric', default='k34', type=str,
                    help='metric to use')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())

if not os.path.exists(args.save):
    os.makedirs(args.save)


model=models.mobilenet_v2(pretrained=True)



cudnn.benchmark = True

print(model)




print('Pre-processing Successful!')
if args.cuda:
     model.cuda()



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data,'val'), transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()
    with torch.no_grad():
        for i, (input, target,index) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var.cuda())
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




#decide how many and which filters to prune at each layer


layer_id = 1
cfg = []
cfg_mask = []
coef=1.996
count=0
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        out_channels = m.weight.data.shape[0]

        if m.kernel_size == (3,3) and m.in_channels==3:
            cfg_mask.append(torch.ones(1))
            cfg.append(1)
            first=True
            continue
        if m.kernel_size == (1,1) and out_channels==1280:
            continue
        if m.kernel_size == (1,1) and count==0:
            count=1


        elif m.kernel_size == (3,3):
            count=0
            if first==True:
                cfg_mask.append(torch.ones(out_channels))
                cfg.append(out_channels)
                first=False
                num_keep_pre=16
                layer_id+=2
                continue

        if layer_id % 3 == 1 or layer_id % 3 == 2:
            weight_copy = m.weight.data.clone().cpu().numpy()
            a=weight_copy.flatten()
            W, p = stats.shapiro(a)
            prune_prob_stage = np.ceil(10 * (W - 0.78) / (coef * 0.22)) / 10


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

            # no prune of depthwise layers
            if m.kernel_size== (3, 3):
                arg_max=arg_max_pre
                num_keep=num_keep_pre






            arg_max_rev = arg_max[::-1][:num_keep]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1


            cfg_mask.append(mask)
            cfg.append(num_keep)
            arg_max_pre=arg_max
            num_keep_pre=num_keep
        layer_id += 1




assert len(cfg) == 34, "Length of cfg variable is not correct."

newmodel=mobilenet_v2(inverted_residual_setting=cfg)


#transfer weights from the pretrained network to the pruned one


start_mask = torch.ones(3)
layer_id_in_cfg = 1
conv_count = 2
count=0
first_layer=False
for [m0, m1] in zip(model.modules(), newmodel.modules()):

    if isinstance(m0, nn.Conv2d):

        if m0.kernel_size == (3, 3) and m0.in_channels == 3:

            m1.weight.data = m0.weight.data.clone()
            first_layer = True
            continue
        elif m0.kernel_size == (1,1) and m0.out_channels==1280:

            m1.weight.data = m0.weight.data.clone()
            continue
        elif m0.kernel_size == (1,1) and count==0:
            count=1
        elif m0.kernel_size == (3,3):
            count=0

        if conv_count % 3 == 1:

            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))

            w = m0.weight.data[idx.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            layer_id_in_cfg += 1
            conv_count += 1
            continue
        if conv_count % 3 == 0:

            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))

            w = m0.weight.data[:, idx.tolist(), :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            layer_id_in_cfg += 1
            continue
        if conv_count % 3 == 2:

            mask1 = cfg_mask[layer_id_in_cfg]
            idx1 = np.squeeze(np.argwhere(np.asarray(mask1.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w = m0.weight.data[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w.clone()
            conv_count += 1
            continue


    elif isinstance(m0, nn.BatchNorm2d):


        assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."

        if conv_count % 3 == 0  :

            mask = cfg_mask[layer_id_in_cfg]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            m1.num_batches_tracked=m0.num_batches_tracked.clone()

            continue
        if conv_count % 3 == 2 and first_layer==False:
            mask = cfg_mask[layer_id_in_cfg-1 ]
            idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if idx.size == 1:
                idx = np.resize(idx, (1,))
            m1.weight.data = m0.weight.data[idx.tolist()].clone()
            m1.bias.data = m0.bias.data[idx.tolist()].clone()
            m1.running_mean = m0.running_mean[idx.tolist()].clone()
            m1.running_var = m0.running_var[idx.tolist()].clone()
            m1.num_batches_tracked = m0.num_batches_tracked.clone()

            continue

        if first_layer==True:
            first_layer=False



        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
        m1.running_mean = m0.running_mean.clone()
        m1.running_var = m0.running_var.clone()
        m1.num_batches_tracked = m0.num_batches_tracked.clone()

    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()
print(layer_id_in_cfg)

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'mobilenetv2_pruned.pth.tar'))
print(newmodel)
newmodel.cuda()





num_parameters1 = sum([param.nelement() for param in model.parameters()])
num_parameters2=  sum([param.nelement() for param in newmodel.parameters()])


print('FLOPs')
flops_std = count_model_param_flops(model, 224)
flops_std1 = count_model_param_flops(newmodel, 224)


print('flops pruned')
print(1-flops_std1/flops_std)
print('params pruned')
print(1-num_parameters2/num_parameters1)


acc=test(model)

print("number of parameters: "+str(num_parameters2))
with open(os.path.join(args.save, "mobilenetv2_pruned.txt"), "w") as fp:
    fp.write("Number of parameters: \n"+str(num_parameters2)+"\n")
    fp.write("Test accuracy: \n"+str(float(acc))+"\n")


