from .cvt import cvt_13_224x224
from .micronet import micronet_m0, micronet_m1, micronet_m2, micronet_m3
from .mobilenext import MobileNeXt_100
from .peleenet import peleenet
from .cycle_mlp import CycleMLP_B1
from .convnext import convnext_tiny
from .cait import cait_XXS24_224


def create_model(model, **kwargs):
    model = eval(model)(**kwargs)
    return model
