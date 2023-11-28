import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader



def parse_args() :
    parser = argparse.ArgumentParser(description="Train options")
    # Model parameters
    parser.add_argument("--model", type=str, default="MLP", help="Model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default=None, help="Scheduler")
    parser.add_argument("--folds", type=int, default=3, help="number of folds")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb-project-name", type=str, help="Use wandb")

    args = parser.parse_args()
    return args

class Weighted_MSE(torch.nn.Module):
    def __init__(self, weights, reduction = "mean"): 
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )

        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        _weights = torch.tensor( [[ self.weights[int(x)] for x in row] for row in targets])
        x = self._reduce((inputs - targets)**2 * _weights)
        return x

    def _reduce(self, inputs):
        if self.reduction == 'mean' :
            return inputs.sum() / inputs.numel()
        elif self.reduction == "sum" :
            return inputs.sum()

def normalize_each_sample(x):
    norm = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
    return norm.float()


def dict_to_list(labels) :
    return torch.tensor([ x for x in labels.values() ]).float()


def prepare_dataloader(dataset: Dataset, batch_size: int, collate_fn=None, shuffle = True, sampler = None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle = shuffle,
        sampler = sampler,
        collate_fn=collate_fn,
    )



def error_per_au_per_intensity(predictions, labels):
    action_units = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']

    total_mse = []
    for au_index, au in enumerate(action_units) :
        pred_per_au = predictions[:, au_index]
        labels_per_au = labels[:, au_index]

        err_per_intensity = []
        for intensity in range(6) :
            mask = labels == intensity
            mse = np.square(labels_per_au[mask] - pred_per_au[mask]).mean()
            err_per_intensity.append(mse)

        total_mse.append(err_per_intensity)

    return total_mse


if __name__ == "__main__" :

    weights = torch.arange(6) + 1

    target = torch.from_numpy(np.random.randint(6, size=(10, 12)))

    predict = torch.from_numpy(np.random.rand(10, 12) * 5)

    wmse = Weighted_MSE(weights=weights)

    a = wmse(predict, target)
    