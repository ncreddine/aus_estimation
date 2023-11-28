from .model import Model_MLP 
from .resnet import ResNet, ResidualBlock
from .pointnet import PointNet, PointNet_loss 


def get_model(name): 
    if name == "MLP" :
        return Model_MLP
    elif name == "resnet" :
        return ResNet
    elif name == "pointnet" :
        return PointNet, PointNet_loss
    else :
       raise NotImplementedError(f"Model {name} not implemented")