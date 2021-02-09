import torch
import torch.nn as nn
from modules import SAB, PMA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def gen_data(batch_size, max_length=10, test=False):
    x = np.random.randint(1, 100, (batch_size, max_length, 5))
    y = np.max(x, axis=1)[:,0]

    return x, y

class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        print("Set Transformer Model Initialized")
        self.enc = nn.Sequential(
            SAB(dim_in=5, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)


def train(model):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss().cuda()
    losses = []
    for _ in range(500):
        x, y = gen_data(batch_size=2 ** 10, max_length=10)
        x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).float().cuda().unsqueeze(1)

        output = model(x)
        #print("output shape: ", output.shape)
        #print("y shape: ", y.shape)
        #raise ValueError()
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

models = [("Set Transformer", SmallSetTransformer())]

for _name, _model in models:
    _losses = train(_model)
    plt.plot(_losses, label=_name)
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Mean Absolute Error")
plt.yscale("log")
plt.show()
