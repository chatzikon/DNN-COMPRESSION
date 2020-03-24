import torch
import torch.nn as nn
from models.resnet import BasicBlock
import numpy as np



class View(nn.Module):
    """
    Reshape data from 4 dimension to 2 dimension
    """

    def forward(self, x):
        assert x.dim() == 2 or x.dim() == 4, "invalid dimension of input {:d}".format(x.dim())
        if x.dim() == 4:
            out = x.view(x.size(0), -1)
        else:
            out = x
        return out



class AuxClassifier(nn.Module):
    """
    define auxiliary classifier:
    AVGPOOLING->FC
    """

    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.linear = nn.Linear(in_channels, num_classes)

        # init params
        self.linear.bias.data.zero_()

    def forward(self, x):
        """
        forward propagation
        """
        out = x.mean(2).mean(2)
        out = self.linear(out)
        return out




def create(model,lr,exp,predefined,n_losses,decomp,arch,depth,dataset):


    if arch=='resnet':
        if depth==56:
            network_block_num=27
        elif depth==110:
            network_block_num = 54
    elif arch=='MobileNet':
        network_block_num = 13


    #predefined positions if intermediate auxliary losses<=2, at the output of the ResNet layers
    if predefined:
        if depth==56:
            if n_losses==1:
                pivot_set=[18]
            elif n_losses==2:
                pivot_set=[9,18]
        elif depth==110:
            if n_losses==1:
                pivot_set=[18]
            if n_losses==2:
                pivot_set=[18,36]

    else:
        pivot_set=cal_pivot(n_losses,network_block_num)
        print(pivot_set)


    pivot_weight,lr_weight= set_loss_weight(pivot_set,exp)


    print(pivot_set)
    print(pivot_weight)
    print(lr_weight)

    if arch=='resnet':

        final_block_count, segments=create_segments_resnet(model,pivot_set)

    elif arch=='MobileNet':

        final_block_count, segments = create_segments_mobilenet(model, pivot_set)






    aux_fc=create_auxiliary_classifiers(segments,model,decomp,dataset)

    model.cuda()
    for i in range(len(segments)):
        segments[i].cuda()


    for i in range(len(aux_fc)):
        aux_fc[i].cuda()

    seg_optimizer, fc_optimizer=create_optimizers(segments,lr,aux_fc)

    return  seg_optimizer,fc_optimizer,final_block_count,segments, aux_fc,pivot_weight,lr_weight





def cal_pivot(n_losses,network_block_num):
    """
    Calculate the inserted layer for additional loss
    """


    num_segments = n_losses + 1

    num_block_per_segment = (network_block_num // num_segments) + 1



    pivot_set = []
    for i in range(num_segments - 1):
        pivot_set.append(min(num_block_per_segment * (i + 1), network_block_num - 1))





    return  pivot_set








def set_loss_weight(pivot_set,exp):
    """
    The weight of the k-th auxiliary loss: gamma_k = \max(0.01, (\frac{L_k}{L_K})^2)
    More details can be found in Section 3.2 in "The Shallow End: Empowering Shallower Deep-Convolutional Networks
    through Auxiliary Outputs": https://arxiv.org/abs/1611.01773.
    """


    base_weight = 0

    lr_weight = torch.zeros(len(pivot_set)).cuda()
    pivot_weight = lr_weight.clone()

    for i in range(len(pivot_set) - 1, -1, -1):
        temp_weight = max(pow(float(pivot_set[i]/pivot_set[-1]), exp), 0.01)

        base_weight += temp_weight
        pivot_weight[i] = temp_weight
        lr_weight[i] = base_weight

    return pivot_weight, lr_weight

def create_segments_resnet(model,pivot_set):


    segments=[]



    net_head = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu)
    shallow_model = nn.Sequential(net_head)





    block_count = 0

    for name, module in model.named_modules():





        if len(name)==8 or len(name)==9:






                if shallow_model is not None:
                    shallow_model.add_module(str(len(shallow_model)), module)
                else:
                    shallow_model = nn.Sequential(module)
                block_count += 1

                # if block_count is equals to pivot_num, then create new segment
                if block_count in pivot_set:
                    segments.append(shallow_model)
                    shallow_model = None

    final_block_count = block_count
    segments.append(shallow_model)














    return  final_block_count,segments



def create_segments_mobilenet(model,pivot_set):


    segments=[]



    net_head = nn.Sequential(
        model.conv1,
        model.bn1)
    shallow_model = nn.Sequential(net_head)





    block_count = 0

    for name, module in model.named_modules():





        if len(name)==8 or len(name)==9:






                if shallow_model is not None:
                    shallow_model.add_module(str(len(shallow_model)), module)
                else:
                    shallow_model = nn.Sequential(module)
                block_count += 1

                # if block_count is equals to pivot_num, then create new segment
                if block_count in pivot_set:
                    segments.append(shallow_model)
                    shallow_model = None

    final_block_count = block_count
    segments.append(shallow_model)














    return  final_block_count,segments





def create_auxiliary_classifiers(segments,model,decomp,dataset):


    if dataset=='cifar10':
        num_classes = 10
    elif dataset=='cifar100':
        num_classes=100
    aux_fc=[]





    for i in range(len(segments) - 1):


        if decomp:
            in_channels = segments[i][-1].conv2[2].out_channels
        else:
            in_channels = segments[i][-1].conv2.out_channels



        assert in_channels != 0, "in_channels is zero"

        aux_fc.append(AuxClassifier(in_channels=in_channels, num_classes=num_classes))



    final_fc = nn.Sequential(*[
        model.avgpool,
        View(),
        model.linear])

    aux_fc.append(final_fc)


    return  aux_fc


def create_optimizers(segments,lr,aux_fc):

    seg_optimizer = []
    fc_optimizer = []

    for i in range(len(segments)):

        temp_optim = []
        # add parameters in segments into optimizer
        # from the i-th optimizer contains [0:i] segments
        for j in range(i + 1):

            temp_optim.append({'params': segments[j].parameters(),
                               'lr': lr})



        # optimizer for segments and fc
        temp_seg_optim = torch.optim.SGD(
            temp_optim,
            momentum=0.9,
            weight_decay=0,
            nesterov=True)

        temp_fc_optim = torch.optim.SGD(
            params=aux_fc[i].parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=0,
            nesterov=True)

        seg_optimizer.append(temp_seg_optim)
        fc_optimizer.append(temp_fc_optim)

    return  seg_optimizer, fc_optimizer







