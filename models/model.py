import torch
import torch.nn as nn
import torch.nn.functional as F



class Model_MLP(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(Model_MLP, self).__init__()

        self.input_fc  = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.hidden_fc = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.output_fc = nn.Linear(256, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)
        h1 = F.relu(self.bn1(self.input_fc(x)))
        h2 = F.relu(self.bn2(self.hidden_fc(h1)))
        y_pred = F.relu(self.output_fc(h2))

        return y_pred



if __name__ == "__main__" :
    model = Model_MLP(input_dim = 478*3, output_dim = 12)

    a = torch.randn(3, 478, 3)

    with torch.no_grad() :
        print(model(a))