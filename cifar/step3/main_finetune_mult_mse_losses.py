from __future__ import print_function
import argparse
import numpy as np
import os
import shutil
import  time
import torch.nn as nn
import torch.nn.functional as F
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

import net_for_multiloss_ft

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--refine', default='../step2/decomposed_models/models_finetuned/resnet56_cifar10/tucker2/1.71x/layer_groups:3/t.pth.tar', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
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
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=56, type=int,
                    help='depth of the neural network')

def seed_everything(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']=str(SEED)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset all parameters
        """

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update parameters
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def _convert_results(top1_accuracy, top1_loss, top5_accuracy):
    """
    Convert tensor list to float list
    :param top1_error: top1_error tensor list
    :param top1_loss:  top1_loss tensor list
    :param top5_error:  top5_error tensor list
    """

    assert isinstance(top1_accuracy, list), "input should be a list"
    length = len(top1_accuracy)
    top1_accuracy_list = []
    top5_accuracy_list = []
    top1_loss_list = []
    for i in range(length):
        top1_accuracy_list.append(top1_accuracy[i].avg)
        top5_accuracy_list.append(top5_accuracy[i].avg)
        top1_loss_list.append(top1_loss[i].avg)
    return top1_accuracy_list, top1_loss_list, top5_accuracy_list


def print_result(epoch, nEpochs, count, iters, data_time, iter_time, error, loss, top5error=None,
                 mode="Train"):
    log_str = "{}: [{:0>3d}|{:0>3d}], Iter: [{:0>3d}|{:0>3d}],  DataTime: {:.4f}, IterTime: {:.4f}, ".format(
        mode, epoch + 1, nEpochs, count, iters,  data_time, iter_time)

    if isinstance(error, list) or isinstance(error, np.ndarray):
        for i in range(len(error)):
            log_str += "Error_{:d}: {:.4f}, Loss_{:d}: {:.4f}, ".format(i, error[i], i, loss[i])
    else:
        log_str += "Error: {:.4f}, Loss: {:.4f}, ".format(error, loss)

    if top5error is not None:
        if isinstance(top5error, list) or isinstance(top5error, np.ndarray):
            for i in range(len(top5error)):
                log_str += " Top5_Error_{:d}: {:.4f}, ".format(i, top5error[i])
        else:
            log_str += " Top5_Error: {:.4f}, ".format(top5error)









def accuracy(outputs, labels, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    :param outputs: the outputs of the model
    :param labels: the ground truth of the data
    :param topk: the list of k in top-k
    :return: accuracy
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)


        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def compute_singlecrop_error(outputs, labels, loss, top5_flag=False):
    """
    Compute singlecrop top-1 and top-5 error
    :param outputs: the output of the model
    :param labels: the ground truth of the data
    :param loss: the loss value of current batch
    :param top5_flag: whether to calculate the top-5 error
    :return: top-1 error list, loss list and top-5 error list
    """

    with torch.no_grad():
        if isinstance(outputs, list):
            top1_loss = []
            top1_accuracy = []
            top5_accuracy = []



            for i in range(len(outputs)):
                top1_acc, top5_acc = accuracy(outputs[i], labels, topk=(1, 5))

                top1_accuracy.append(top1_acc)
                top5_accuracy.append(top5_acc)
                top1_loss.append(loss[i].item())


        else:
            top1_acc, top5_acc = accuracy(outputs, labels, topk=(1, 5))

            top1_accuracy=top1_acc
            top5_accuracy=top5_acc
            top1_loss = loss.item()

        if top5_flag:
            return top1_accuracy, top1_loss, top5_accuracy
        else:
            return top1_accuracy, top1_loss


def auxnet_forward(images, labels, segments,auxiliary_fc,output_temp, mse_factor,train):
    """
    Forward propagation fot auxnet
    """

    outputs = []
    temp_input = images
    losses = []





    for i in range(len(segments)):
        # forward

        temp_output = segments[i](temp_input)
        fcs_output = auxiliary_fc[i](temp_output)
        outputs.append(fcs_output)
        if labels is not None:
            if train :
                #last MSE loss
                if i==len(segments)-1:

                    auxiliary_output = F.cross_entropy(fcs_output, labels) + mse_factor * F.mse_loss(fcs_output, output_temp[i])
                #intermediate MSE losses
                else:
                    auxiliary_output = F.mse_loss(temp_output, output_temp[i])

            else:
                auxiliary_output = F.cross_entropy(fcs_output, labels)
            losses.append(auxiliary_output)

        temp_input = temp_output


    return outputs, losses

def auxnet_backward_for_loss_i(loss, i,segment_optimizer,fc_optimizer,pivot_weight):
    """
    Backward propagation for the i-th loss
    :param loss: the i-th loss
    :param i: which one to perform backward propagation
    """

    segment_optimizer[i].zero_grad()
    fc_optimizer[i].zero_grad()




    if i < len(segment_optimizer) - 1:



        loss.backward(retain_graph=True)

        #reduce the impact of the intermediate loss using the equation 14 of the paper
        for param_group in segment_optimizer[i].param_groups:
            for p in param_group['params']:
                if p.grad is None:
                    continue
                p.grad.data.mul_(p.new([pivot_weight[i]]))
    else:
        loss.backward(retain_graph=True)


    fc_optimizer[i].step()
    segment_optimizer[i].step()


def train(train_loader,epoch,output_orig, mse_factor,segments,segment_optimizer,fc_optimizer,auxiliary_fc,pivot_weight,lr_weight,lr):

    top1_accuracy = []
    top5_accuracy = []
    top1_loss = []

    for i in range(len(segment_optimizer)):
        segments[i].train()
        auxiliary_fc[i].train()
        top1_accuracy.append( AverageMeter())
        top5_accuracy.append( AverageMeter())
        top1_loss.append( AverageMeter())


    train = True




    for batch_idx, (data, target,index) in enumerate(train_loader):




        data, target= data.cuda(), target.cuda()

        # forward
        output_temp={}
        for i in range(len(output_orig)):
            output_temp[i]=torch.from_numpy(output_orig[i][index.cpu().numpy(), :]).float().cuda()


        outputs, losses = auxnet_forward(data, target,segments,auxiliary_fc,output_temp, mse_factor,train)

        # backward
        for j in range(len(segment_optimizer)):
            auxnet_backward_for_loss_i(losses[j], j,segment_optimizer,fc_optimizer,pivot_weight)



        single_accuracy, single_loss, single5_accuracy = compute_singlecrop_error(
            outputs=outputs, labels=target,
            loss=losses, top5_flag=True)

        for j in range(len(segment_optimizer)):
            top1_accuracy[j].update(single_accuracy[j], data.size(0))
            top5_accuracy[j].update(single5_accuracy[j], data.size(0))
            top1_loss[j].update(single_loss[j], data.size(0))


        log_interval = 100





        if (batch_idx+1) % log_interval == 0:
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}, Accuracy: {} \n'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_loader.sampler),
                           100. * batch_idx / len(train_loader), top1_loss[len(top1_loss)-1].avg, top1_accuracy[len(top1_accuracy)-1].avg))

    top1_accuracy_list, top1_loss_list, top5_error_list = _convert_results(
        top1_accuracy=top1_accuracy, top1_loss=top1_loss, top5_accuracy=top5_accuracy)



    return top1_accuracy_list, top1_loss_list, top5_error_list






def test(test_loader,segments,auxiliary_fc,mse_factor):



    top1_accuracy = []
    top5_accuracy = []
    top1_loss = []
    num_segments = len(segments)
    for i in range(num_segments):
        segments[i].eval()
        auxiliary_fc[i].eval()
        top1_accuracy.append(AverageMeter())
        top5_accuracy.append(AverageMeter())
        top1_loss.append(AverageMeter())




    output_temp=[]
    train=False



    with torch.no_grad():
        for data, target,index in test_loader:

            data, target = data.cuda(), target.cuda()

            outputs, losses = auxnet_forward(data, target, segments, auxiliary_fc, output_temp, mse_factor,train)
            # compute loss and error rate
            single_accuracy, single_loss, single5_accuracy = compute_singlecrop_error(
                outputs=outputs, labels=target,
                loss=losses, top5_flag=True)



            for j in range(num_segments):
                top1_accuracy[j].update(single_accuracy[j], data.size(0))
                top5_accuracy[j].update(single5_accuracy[j], data.size(0))
                top1_loss[j].update(single_loss[j], data.size(0))







    top1_accuracy_list, top1_loss_list, top5_accuracy_list = _convert_results(
    top1_accuracy=top1_accuracy, top1_loss=top1_loss, top5_accuracy=top5_accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}\n'.format(
        top1_loss_list[len(top1_loss_list) - 1], top1_accuracy_list[len(top1_accuracy_list) - 1], len(test_loader.sampler)))




    return top1_accuracy_list, top1_loss_list, top5_accuracy_list



def test_orig(segments, aux_fc_init, test_loader,dataset):
    print(segments)

    for i in range(len(segments)):
        segments[i].eval()
        aux_fc_init[i].eval()





    output={}
    temp_input=torch.zeros(test_loader.batch_size,3,32,32).cuda()


    for i in range(len(segments)):
        # forward

        temp_output = segments[i](temp_input)
        if i == len(segments) - 1:
            if dataset=='cifar10':
                output[i] = np.zeros((45000, 10))
            elif dataset=='cifar100':
                output[i] = np.zeros((45000, 100))
        else:
            output[i] = np.zeros((45000, temp_output.size()[1], temp_output.size()[2], temp_output.size()[3]))
        temp_input = temp_output





    inds=np.zeros((45000,),dtype=int)



    k=0

    count=0
    with torch.no_grad():

        for datat, target,index in test_loader:




            datat, target = datat.cuda(), target.cuda()
            datat, target= Variable(datat), Variable(target)
            temp_input=datat


            for i in range(len(segments)):
                # forward



                temp_output = segments[i](temp_input)




                if i == len(segments) - 1:



                    fcs_output = aux_fc_init[i](temp_output)
                    output[i][k:k + datat.shape[0], :] = fcs_output.data.cpu().numpy()

                else:
                    output[i][k:k + datat.shape[0], :, :, :] = temp_output.data.cpu().numpy()

                temp_input = temp_output
            inds[k:k + datat.shape[0]] = index.cpu().numpy()
            k = k + datat.shape[0]

            count += 1




    return output, inds

def save_checkpoint(state, is_best,counter, filepath):
    torch.save(state, os.path.join(filepath, 'checkpointB.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpointB.pth.tar'), os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(state['best_prec1'])+'.pth.tar'))


def save_checkpoint_1(state,counter, filepath):
    torch.save(state, os.path.join(filepath, 'modelB_best_test_set_acc_'+str(counter)+'_'+str(state['best_prec1'])+'.pth.tar'))


def load_checkpoint(best,counter,filepath):
    if os.path.isfile(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar')):
        print("=> loading checkpoint '{}'".format(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar')))
        checkpoint = torch.load(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar'))
        print("=> loaded checkpoint '{}'  Prec1: {:f}".format(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar'), best))
    else:
        print("=> no checkpoint found at '{}'".format(os.path.join(filepath, 'modelB_best_test_acc_'+str(counter)+'_'+str(best)+'.pth.tar')))
    return checkpoint


def main(savedir,resume_orig,mse_factor,exp,predefined,n_losses):






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
    model=torch.load(args.refine)

    #sometimes there is a problem with AvgPool2d of the loaded model, if this problem occur, uncomment these lines
    # if 'resnet' in arch:
    #     model.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

    model.cuda()

    print(model)

    best_prec1 = 0.



    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            auxiliary_fc_state = checkpoint["auxiliary_fc"]
            segment_optimizer_state = checkpoint["optimizer_seg"]
            fc_optimizer_state = checkpoint["optimizer_fc"]
            epoch = checkpoint['epoch']
            args.start_epoch=epoch
            print(epoch)
            print("=> loaded checkpoint '{}'  Prec1: {:f}"
                  .format(args.resume, best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # load the baseline pretrained network
    if resume_orig:
        if os.path.isfile(resume_orig):
            print("=> loading checkpoint '{}'".format(resume_orig))
            if 'MobileNet' in arch:
                model_init = models.__dict__[args.arch](num_classes=10)
            else:
                model_init = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
            checkpoint = torch.load(resume_orig)
            model_init.load_state_dict(checkpoint['state_dict'])
            best_prect = checkpoint['best_prec1']
            model_init.eval()
            print("=> loaded checkpoint '{}'  Prec1: {:f}"
                  .format(args.resume, best_prect))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))






    #function which which splits the compressed model in different segments, each segments having an auxiliary mse loss at its output
    segment_optimizer, fc_optimizer, final_block_count, segments,auxiliary_fc,pivot_weight, lr_weight=net_for_multiloss_ft.create(model,args.lr,exp,predefined,n_losses,True,arch,depth,args.dataset)











    if args.resume:
        for i in range(len(auxiliary_fc_state)):
            auxiliary_fc[i].load_state_dict(auxiliary_fc_state[i])
            fc_optimizer[i].load_state_dict(fc_optimizer_state[i])
            segment_optimizer[i].load_state_dict(segment_optimizer_state[i])









    segment_scheduler=[]
    fc_scheduler = []


    #assign a scheduler and an optimizer at each network segment


    for i in range(len(segment_optimizer)):
        segment_scheduler.append(MultiStepLR(segment_optimizer[i], milestones=[80,120], gamma=0.1))


    for i in range(len(fc_optimizer)):
        fc_scheduler.append(MultiStepLR(fc_optimizer[i], milestones=[80,120], gamma=0.1))







    print(model)
    print(model_init)

    model_init.cuda()


    #function which which splits the baseline model in different segments, each segments having an auxiliary mse loss at its output
    _, _, _, segments_init, auxiliary_fc_init, _, _ = net_for_multiloss_ft.create(model_init, args.lr, exp, predefined, n_losses,False,arch,depth,args.dataset)




    #extract the softmax layer values from the baseline network
    print('test orig')
    output,inds = test_orig(segments_init, auxiliary_fc_init, train_loader,args.dataset)


    #array to store the output values of the baseline network
    outputf = {}
    temp_input = torch.zeros(test_loader.batch_size, 3, 32, 32).cuda()


    #the softmax layer values of the baseline network is assigned to an array in order to get compared with the compressed network output at the MSE loss


    for i in range(len(segments_init)):

        temp_output = segments_init[i](temp_input)
        if i == len(segments_init) - 1:
            if args.dataset=='cifar10':
                outputf[i] = np.zeros((50000, 10))
            elif args.dataset=='cifar100':
                outputf[i] = np.zeros((50000, 100))
        else:
            outputf[i] = np.zeros((50000, temp_output.size()[1], temp_output.size()[2], temp_output.size()[3]))
        temp_input = temp_output


    #the intermediate output layer values of the baseline network is assigned to an array in order to get compared with the compressed network output at the MSE loss

    for i in range(len(segments_init)):
        for j in range(len(inds)):
            if i == len(segments_init) - 1:
                outputf[i][inds[j],:]=output[i][j,:]
            else:
                if output[i].shape!=output[i].shape:

                    outputf[i][inds[j], :,:,:] = output[i][j, :,:,:]

    print('test 1')
    val_error, val_loss, val5_error = test(test_loader, segments_init, auxiliary_fc_init, mse_factor)


    print(val_error)
    print(val_loss)
    print(val5_error)
    #
    print('test 2')
    val_error, val_loss, val5_error = test(test_loader, segments, auxiliary_fc,  mse_factor)

    print(val_error)
    print(val_loss)
    print(val5_error)









    for epoch in range(args.start_epoch, args.epochs):

        print('LR')
        for i in range(len(segment_optimizer)):
            print(segment_optimizer[i].param_groups[0]['lr'])

        for i in range(len(fc_optimizer)):
            print(fc_optimizer[i].param_groups[0]['lr'])


        start_time = time.time()
        _,_,_=train(train_loader, epoch,outputf,mse_factor,segments,segment_optimizer,fc_optimizer,auxiliary_fc,pivot_weight,lr_weight,args.lr)




        print('LR')
        for i in range(len(segment_optimizer)):
            segment_scheduler[i].step(epoch)
            print(segment_optimizer[i].param_groups[0]['lr'])

        for i in range(len(fc_optimizer)):
            fc_scheduler[i].step(epoch)
            print(fc_optimizer[i].param_groups[0]['lr'])




        val_accuracy_list, val_loss_list, val5_accuracy_list = test(valid_loader,segments, auxiliary_fc,mse_factor)
        print(val_accuracy_list[-1])
        print(best_prec1)


        # save model and checkpoint
        best_flag = False
        if  val_accuracy_list[-1]>=best_prec1:
            best_prec1 = val_accuracy_list[-1]
            best_flag = True
        print(best_flag)



        seg_opt_state=[]
        fc_opt_state=[]
        auxiliary_fc_state=[]


        for i in range(len(segment_optimizer)):
            seg_opt_state.append(segment_optimizer[i].state_dict())
        for i in range(len(fc_optimizer)):
            fc_opt_state.append(fc_optimizer[i].state_dict())
            auxiliary_fc_state.append(auxiliary_fc[i].state_dict())

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer_seg': seg_opt_state,
            'optimizer_fc': fc_opt_state,
            'auxiliary_fc': auxiliary_fc_state
        }, best_flag, counter, filepath=args.save)
        elapsed_time = time.time() - start_time
        print(elapsed_time)
        if epoch==args.epochs-1:
            test_accuracy_list,_,_ = test(test_loader,segments, auxiliary_fc,epoch,args.epochs,mse_factor)
            prec_T = test_accuracy_list[-1]
            save_checkpoint_1({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': prec_T,
            'optimizer_seg': seg_opt_state,
            'optimizer_fc': fc_opt_state,
            'auxiliary_fc': auxiliary_fc_state
            }, counter, filepath=args.save)


    checkpoint_t=load_checkpoint(best_prec1,counter,args.save)
    model.load_state_dict(checkpoint_t['state_dict'])
    segment_optimizer, fc_optimizer, final_block_count, segments, auxiliary_fc, pivot_weight,lr_weight = net_for_multiloss_ft.create(model,
                                                                                                                 args.lr,exp,predefined,n_losses,True,arch,depth,args.dataset)


    auxiliary_fc_state = checkpoint_t["auxiliary_fc"]
    for i in range(len(auxiliary_fc_state)):
        auxiliary_fc[i].load_state_dict(auxiliary_fc_state[i])


    test_accuracy_list, test_loss_list, test5_accuracy_list = test(test_loader,segments, auxiliary_fc,mse_factor)
    prec_f = test_accuracy_list[-1]
    print(prec_f)


    best_prec1 = prec_f
    save_checkpoint_1({
        'epoch': args.epochs + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'segment_optimizer': segment_optimizer_state,
        'fc_optimizer': fc_optimizer_state,
        'auxiliary_fc': auxiliary_fc_state
    },counter, filepath=args.save)






if __name__ == '__main__':



    arch = 'resnet'
    depth = 56
    counter = 2
    resume_orig = '../step1/cifar10/pretrained_models/resnet_model_best_acc_0.9264.pth.tar'

    path = './logd' + str(counter)
    if not os.path.exists(path):
        os.makedirs(path)
    savedir = path

    #the decay rate v (equation 14 of the paper)
    exp = 3
    predefined = True
    n_losses = 2
    mse_factor = 5

    main(savedir, resume_orig, mse_factor, exp,  predefined, n_losses)

