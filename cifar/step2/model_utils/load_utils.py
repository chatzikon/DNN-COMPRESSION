import torch
import sys
sys.path.insert(0, "../step1/cifar10/")
import models




PWD = './'
PATH_TO_PRETRAINED = '{}logs'.format(PWD)
SAVE_ROOT = '{}decomposed_models'.format(PWD)



def load_model(MODEL_NAME):



    if MODEL_NAME=='resnet56_cifar10':

        PATH_TO_MODEL_init = '../step1/cifar10/pretrained_models/resnet_model_best_acc_0.9264.pth.tar'
        checkpoint_init = torch.load(PATH_TO_MODEL_init)


        model_init=models.resnet(depth=56,dataset='cifar10')
        model_init.load_state_dict(checkpoint_init['state_dict'])




        PATH_TO_MODEL = '../step1/cifar10/pruned_models/res56_norm.pth.tar'
        checkpoint = torch.load(PATH_TO_MODEL)

        model = models.resnet(depth=56, cfg=checkpoint['cfg'],dataset='cifar10')
        model.load_state_dict(checkpoint['state_dict'])


    elif MODEL_NAME=='resnet56_cifar100':

        PATH_TO_MODEL_init = '../step1/cifar100/pretrained_models/resnet56_c100_0_7085.pth.tar'
        checkpoint_init = torch.load(PATH_TO_MODEL_init)

        model_init= models.resnet(depth=56,dataset='cifar100')
        model_init.load_state_dict(checkpoint_init['state_dict'])




        PATH_TO_MODEL='../step1/cifar100/pruned_models/res56_c100.pth.tar'
        checkpoint = torch.load(PATH_TO_MODEL)

        model = models.resnet(depth=56, cfg=checkpoint['cfg'],dataset='cifar100')
        model.load_state_dict(checkpoint['state_dict'])

    elif MODEL_NAME=='resnet110_cifar10':

        PATH_TO_MODEL_init = '../step1/cifar10/pretrained_models/resnet110_model_best_0.9322.pth.tar'
        checkpoint_init = torch.load(PATH_TO_MODEL_init)

        model_init = models.resnet(depth=110,dataset='cifar10')
        model_init.load_state_dict(checkpoint_init['state_dict'])




        PATH_TO_MODEL = '../step1/cifar10/pruned_models/res110_norm.pth.tar'
        checkpoint = torch.load(PATH_TO_MODEL)

        model = models.resnet(depth=110, cfg=checkpoint['cfg'],dataset='cifar10')
        model.load_state_dict(checkpoint['state_dict'])


    elif MODEL_NAME=='resnet110_cifar100':

        PATH_TO_MODEL_init = '../step1/cifar100/pretrained_models/resnet110_c100_0.7241.pth.tar'
        checkpoint_init = torch.load(PATH_TO_MODEL_init)

        model_init = models.resnet(depth=110,dataset='cifar100')
        model_init.load_state_dict(checkpoint_init['state_dict'])




        PATH_TO_MODEL = '../step1/cifar100/pruned_models/res110_c100.pth.tar'
        checkpoint = torch.load(PATH_TO_MODEL)

        model = models.resnet(depth=110, cfg=checkpoint['cfg'],dataset='cifar100')
        model.load_state_dict(checkpoint['state_dict'])



    elif MODEL_NAME =='mobilenetv1_cifar10':

        PATH_TO_MODEL_init = '../step1/cifar10/pretrained_models/mobilenet_0.936.pth.tar'
        checkpoint_init = torch.load(PATH_TO_MODEL_init)

        model_init = models.mobilenet.MobileNet(num_classes=10)
        model_init.load_state_dict(checkpoint_init['state_dict'])

        PATH_TO_MODEL = '../step1/cifar10/pruned_models/mobilenet.pth.tar'
        checkpoint = torch.load(PATH_TO_MODEL)

        model = models.mobilenet.MobileNet(num_classes=10, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])


    elif MODEL_NAME =='mobilenetv1_cifar100':

        PATH_TO_MODEL_init = '../step1/cifar100/pretrained_models/mobilenet_c100_0.7631.pth.tar'
        checkpoint_init = torch.load(PATH_TO_MODEL_init)

        model_init = models.mobilenet.MobileNet(num_classes=100)
        model_init.load_state_dict(checkpoint_init['state_dict'])

        PATH_TO_MODEL = '../step1/cifar100/pruned_models/mobilenet_c100.pth.tar'

        checkpoint = torch.load(PATH_TO_MODEL)
        model = models.mobilenet.MobileNet(num_classes=100, cfg=checkpoint['cfg'])

        model.load_state_dict(checkpoint['state_dict'])

    elif MODEL_NAME =='mobilenetv2_cifar10':

        PATH_TO_MODEL_init = '../step1/cifar10/pretrained_models/mobilenetv2_0.934.pth.tar'
        checkpoint_init = torch.load(PATH_TO_MODEL_init)

        model_init = models.mobilenetv2.MobileNetV2(num_classes=10)
        model_init.load_state_dict(checkpoint_init['state_dict'])

        PATH_TO_MODEL = '../step1/cifar10/pruned_models/mobilenetv2.pth.tar'
        checkpoint = torch.load(PATH_TO_MODEL)

        model=models.mobilenetv2.MobileNetV2(num_classes=10, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])


    elif MODEL_NAME == 'mobilenetv2_cifar100':

        PATH_TO_MODEL_init = '../step1/cifar100/pretrained_models/mobilenetv2_c100_0.7501.tar'
        checkpoint_init = torch.load(PATH_TO_MODEL_init)

        model_init = models.mobilenetv2.MobileNetV2(num_classes=100)
        model_init.load_state_dict(checkpoint_init['state_dict'])

        PATH_TO_MODEL = '../step1/cifar100/pruned_models/mobilenetv2_c100.pth.tar'
        checkpoint = torch.load(PATH_TO_MODEL)

        model = models.mobilenetv2.MobileNetV2(num_classes=100, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])







    return model_init,model
