import torch
import torch.nn as nn
from modules import SAB, PMA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train(config, training_loader):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.L1Loss().cuda()
    # TODO: add classweights
    criterion = nn.CrossEntropyLoss()
    losses = []
    for x, y in training_loader:
        x, y = x.to(device), y.to(device).long()

        output = model(x)
        #print("output shape: ", output.shape)
        #print("y shape: ", y.shape)
        #raise ValueError()
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (i+1) % config.print_every==0:
            print("[==========================]")
            print("Current Iteration: {}".format(i+1))
            print('Total Loss: %.4f' % np.mean(total_l[-100:]))
            print('classification Loss: %.4f' % np.mean(class_l[-100:]))
            print('Segmentation Loss: %.4f' % np.mean(seg_l[-100:]))
            print("Learning Rate: ", optimizer.param_groups[0]['lr'])
            if config.validate: 
                dice = test_net(net, val_loader,val_num=config.val_num)
                save_model(net,dir_checkpoint + model_name + str(dice) + '_{}.pth'.format(i), save_model=save_cp)             
                print("checkpoint saved")

            net.train()
    return losses

models = [("Set Transformer", )]

for _name, _model in models:
    _losses = train(_model)
    plt.plot(_losses, label=_name)
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Mean Absolute Error")
plt.yscale("log")
plt.show()

def main():

    config = set_transformers_config("testing phase")

    model = SmallSetTransformer()
    # Load Data
    training_data = get_transformer_data(config)
    val_data = get_transformer_val_data(config)

    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False)

    model = nn.DataParallel(model).to(device)
    if config.model_checkpoint is not None:
        with open(config.model_checkpoint, 'rb') as f:
                state_dict = torch.load(f)
                model.load_state_dict(state_dict)
                print("Model Loaded!")
            
    model.train()
    train(model, training_loader, training_data)

if __name__ == '__main__':
    main()
