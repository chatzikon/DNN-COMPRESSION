import torch
from torchvision import models
import sys
sys.path.insert(0, "../step1/")
from mobilenetv2_imagenet import mobilenet_v2





PWD = './'
SAVE_ROOT = '{}decomposed_models'.format(PWD)



def load_model(MODEL_NAME):
    if MODEL_NAME == 'mobilenetv2_imagenet':
        model_init = models.mobilenet_v2(pretrained=True)

        PATH_TO_MODEL = '../step1/mobilenetv2_pruned.pth.tar'

        checkpoint = torch.load(PATH_TO_MODEL)
        model = mobilenet_v2(inverted_residual_setting=checkpoint['cfg'])

        model.load_state_dict(checkpoint['state_dict'])




    return model_init,model
