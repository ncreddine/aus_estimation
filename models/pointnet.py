import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class PointNet(nn.Module):
    def __init__(self, k=12, normal_channel=True):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.fc3(x))
        # x = F.relu(x)
        # x = F.log_softmax(x, dim=1)
        return x, trans_feat

class PointNet_loss(torch.nn.Module):
    def __init__(self, weights, mat_diff_loss_scale=0.001, reduction = "mean"):
        super(PointNet_loss, self).__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        self.weights = weights
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.reduction = reduction

    def mse(self, inputs, targets):
        _weights = torch.tensor( [[ self.weights[int(x)] for x in row] for row in targets])
        _weights = _weights.cuda()
        x = self._reduce((inputs - targets)**2 * _weights)
        return x

    def _reduce(self, inputs):
        if self.reduction == 'mean' :
            return inputs.sum() / inputs.numel()
        elif self.reduction == "sum" :
            return inputs.sum()

    def forward(self, pred, target, trans_feat):
        loss = self.mse(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss



# class FocalLoss(torch.nn.Module) :
#     def __init__(self, )