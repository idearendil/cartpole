"""
Neural network file which includes saving, loading, and resetting network
"""

import os
import torch
from torch import nn


class Network(nn.Module):
    """
    Neural network class(DNN).
    """

    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.linear1 = nn.Linear(4, 256, bias=True)
        self.linear2 = nn.Linear(256, 128, bias=True)
        self.linear3 = nn.Linear(128, 64, bias=True)
        self.linear4 = nn.Linear(64, 1, bias=True)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.linear4.weight)

    def forward(self, info: torch.FloatTensor):
        """
        Neural network forward function.
        """
        out = self.layer1(self.linear1(info))
        out = self.layer2(self.linear2(out))
        out = self.layer3(self.linear3(out))
        out = self.linear4(out)
        return out


def save_network(model, saved_dir='./model', file_name='network.pt'):
    os.makedirs(saved_dir, exist_ok=True)
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)


def load_value_network(device) -> Network:
    model_path = './model/value_network_cnn.pt'
    model = Network().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


def data_preprocessing(data, device):
    return data


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    model = Network().to(device)
    save_network(model)