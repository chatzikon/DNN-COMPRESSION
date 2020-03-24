from __future__ import print_function
import argparse
import numpy as np
import os
import shutil
import torchnet as tnt
import  time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torch
import random


import sys
sys.path.insert(0, "../step1/cifar100/")
from data_loader_100 import get_train_valid_loader, get_test_loader

sys.path.insert(0, "../step1/cifar10/")
from data_loader import get_train_valid_loader, get_test_loader
import models


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--refine', default='../step1/cifar10/pruned_models/mobilenet.pth.tar', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=140, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs13', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='MobileNet', type=str,
                    help='architecture to use')


def seed_everything(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']=str(SEED)




def train(model,optimizer,train_loader,epoch,output_orig,mse_factor):
    model.train()
    avg_loss = tnt.meter.AverageValueMeter()
    train_acc = 0.
    for batch_idx, (data, target,index) in enumerate(train_loader):

        data, target= data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        output_temp = torch.from_numpy(output_orig[index.cpu().numpy(), :]).float().cuda()



        loss = F.cross_entropy(output, target).cuda()+mse_factor*F.mse_loss(output,output_temp).cuda()


        avg_loss.add(loss.item())
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        log_interval=100
        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.sampler),
                100. *(batch_idx*len(target)) / len(train_loader), loss.item(), train_acc, (batch_idx+1) * len(data),
                100. * float(train_acc) / ((batch_idx+1) * len(data))))

    return loss.item(), float(train_acc) / len(train_loader.sampler)




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


def test_orig(model,test_loader,dataset):
    model.eval()

    if dataset == 'cifar10':
        output = np.zeros((45000, 10))
    elif dataset == 'cifar100':
        output = np.zeros((45000, 100))

    inds = np.zeros((45000,), dtype=int)

    k=0

    with torch.no_grad():
        for datat, target,index in test_loader:



            datat, target = datat.cuda(), target.cuda()
            datat, target= Variable(datat), Variable(target)
            b1=model(datat)
            output[k:k+datat.shape[0],:]=b1.data.cpu().numpy()
            inds[k:k+datat.shape[0]]=index.cpu().numpy()
            k=k+datat.shape[0]




    return output,inds

def save_checkpoint(state, is_best,counter, filepath):
    torch.save(state, os.path.join(filepath, 'checkpointB.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpointB.pth.tar'), os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(state['best_prec1'])+'.pth.tar'))


def save_checkpoint_1(state, counter, filepath):
    torch.save(state, os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(state['best_prec1'])+'.pth.tar'))


def load_checkpoint(best,counter,filepath):
    if os.path.isfile(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar')):
        print("=> loading checkpoint '{}'".format(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar')))
        checkpoint = torch.load(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar'))
        print("=> loaded checkpoint '{}'  Prec1: {:f}".format(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar'), best))
    else:
        print("=> no checkpoint found at '{}'".format(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar')))
    return checkpoint






def main(savedir,resume_orig,mse_factor):



    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save=savedir





    seed_everything(args.seed)

    if args.dataset == 'cifar10':
        train_loader, valid_loader =get_train_valid_loader('/home/chatziko/PyCharm Projects/dsdimplem/github/binary-wide-resnet-master/cifar10',
                           args.batch_size,
                           augment=True,
                           random_seed=1,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True)



        test_loader = get_test_loader('/home/chatziko/PyCharm Projects/dsdimplem/github/binary-wide-resnet-master/cifar10',
                    args.batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True)
    elif args.dataset == 'cifar100':
        train_loader, valid_loader = get_train_valid_loader100('/home/chatziko/PyCharm Projects/dsdimplem/github/binary-wide-resnet-master/cifar100',
            args.batch_size,
            augment=True,
            random_seed=1,
            valid_size=0.1,
            shuffle=True,
            num_workers=1,
            pin_memory=True)

        test_loader = get_test_loader100(
            '/home/chatziko/PyCharm Projects/dsdimplem/github/binary-wide-resnet-master/cifar100',
            args.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True)










    #load the compressed network
    if args.refine:
        checkpoint = torch.load(args.refine)

        if 'MobileNet' in args.arch:
            if args.dataset == 'cifar10':
                model = models.__dict__[args.arch](num_classes=10,cfg=checkpoint['cfg'])
            elif args.dataset == 'cifar100':
                model = models.__dict__[args.arch](num_classes=100,cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])


    if args.arch=='resnet_mult':
        model.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)


    print(model)
    model.cuda()

    optimizer = optim.SGD(model.parameters(), momentum=args.momentum, lr=args.lr,  weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)



    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(optimizer.param_groups[0]['lr'])

            epoch = checkpoint['epoch']
            args.start_epoch=epoch
            print(epoch)


            print("=> loaded checkpoint '{}'  Prec1: {:f}"
                  .format(args.resume, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print(resume_orig)


    #load the baseline pretrained network
    if resume_orig:
        if os.path.isfile(resume_orig):
            print("=> loading checkpoint '{}'".format(resume_orig))
            if 'MobileNet' in args.arch:
                if args.dataset=='cifar10':
                    model_init = models.__dict__[args.arch](num_classes=10)
                elif args.dataset=='cifar100':
                    model_init = models.__dict__[args.arch](num_classes=100)
            else:
                model_init = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, )
            checkpoint1 = torch.load(resume_orig)
            best_prec1 = checkpoint1['best_prec1']
            model_init.load_state_dict(checkpoint1['state_dict'])
            print("=> loaded checkpoint '{}'  Prec1: {:f}"
                  .format(resume_orig, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(resume_orig))





    print(model_init)



    model_init.cuda()



    #extract the softmax layer values from the baseline network
    output,inds = test_orig(model_init, train_loader,args.dataset)

    if args.dataset == 'cifar10':
        output_base= np.zeros((50000, 10))
    elif args.dataset == 'cifar100':
        output_base = np.zeros((50000, 100))

    temp, _ = test(model, test_loader)
    print(float(temp))


    temp, _ = test(model_init, test_loader)
    print(float(temp))


    #the softmax layer values of the baseline network is assigned to an array in order to get compared with the compressed network output at the MSE loss
    for i in range(len(inds)):
        output_base[inds[i],:]=output[i,:]







    best_prec1 = 0.



    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        train_loss, train_acc=train(model, optimizer, train_loader, epoch,output_base,mse_factor)
        _=float(train_loss)
        print(train_acc)
        scheduler.step(epoch)
        print('LR')
        print(optimizer.param_groups[0]['lr'])
        prec1, valid_loss = test(model,valid_loader)
        prec1=float(prec1)
        _=float(valid_loss)
        print(prec1)
        print(best_prec1)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(is_best)


        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.seed, filepath=args.save)
        elapsed_time = time.time() - start_time
        print(elapsed_time)
        if epoch == args.epochs - 1:
            prec_T, _ = test(model, test_loader)
            prec_T = float(prec_T)
            save_checkpoint_1({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': prec_T,
                'optimizer': optimizer.state_dict(),
            }, args.seed, filepath=args.save)


    checkpoint_t=load_checkpoint(best_prec1,args.seed,args.save)
    model.load_state_dict(checkpoint_t['state_dict'])
    prec_f, _ = test(model, test_loader)
    prec_f = float(prec_f)
    best_prec1 = prec_f
    save_checkpoint_1({
        'epoch': args.epochs + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    },  args.seed, filepath=args.save)






if __name__ == '__main__':

    counter=1
    resume_orig = '../step1/cifar10/pretrained_models/mobilenet_0.936.pth.tar'
    path='./logd' + str(counter)
    if not os.path.exists(path):
        os.makedirs(path)
    savedir = path
    mse_factor = 1
    main(savedir,  resume_orig, mse_factor)







