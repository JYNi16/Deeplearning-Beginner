import torch
from torch import nn, optim
from model import *

def get_optim(model_ft, feature_extract):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    return optimizer_ft
