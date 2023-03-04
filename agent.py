"""
CartpoleAgent class file
"""


import numpy as np
import torch
from torch import nn
from replay_buffer import ReplayBuffer
import dnn
from exploration import boltzmann


class CartpoleAgent():
    """
    Class of cartpole playing agent.
    This class includes updating delayed_network from now_network,
    choosing an action from now_network,
    and training now_network.
    """
    def __init__(self,
                 device,
                 gamma,
                 learning_rate,
                 tau,
                 batch_num,
                 batch_size,
                 replay_buffer_size,
                 is_new=True):
        self.device = device
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_num = batch_num
        self.batch_size = batch_size
        if is_new:
            dnn.reset_network(device=device, model_name="delayed_network")
            dnn.reset_network(device=device, model_name="now_network")
        self.delayed_network = dnn.load_network(device=device,
                                                model_name="delayed_network")
        self.now_network = dnn.load_network(device=device,
                                            model_name="now_network")
        self.replay_buffer = ReplayBuffer(device=device,
                                          max_size=replay_buffer_size)

    def delayed_network_update(self):
        """
        Update delayed_network.
        """
        dnn.save_network(self.now_network, "delayed_network")
        self.delayed_network = dnn.load_network(self.device, "delayed_network")

    def __call__(self, observation):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.array(observation)
                                          ).unsqueeze(dim=0).to(self.device)
            self.now_network.eval()
            value_tensor = torch.softmax(self.now_network.forward(obs_tensor),
                                         dim=1)[0]
            return boltzmann((0, 1),
                             value_tensor.cpu().numpy(),
                             tau=self.tau)

    def test(self, observation):
        """
        Select an action in test mode(only exploitation mode).
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.array(observation)
                                          ).unsqueeze(dim=0).to(self.device)
            self.now_network.eval()
            value_tensor = torch.softmax(self.now_network.forward(obs_tensor),
                                         dim=1)[0]
            return np.argmax(value_tensor.cpu().numpy())

    def train(self):
        """
        Train now_network with the data from replay_buffer.
        """

        if self.replay_buffer.size() < 300:
            return

        # print(f'train - learning_rate: {self.learning_rate}, tau: {
        # self.tau}')

        self.delayed_network.eval()
        self.now_network.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.now_network.parameters(),
                                    lr=self.learning_rate)

        for _ in range(self.batch_num):

            with torch.no_grad():
                batch_data = self.replay_buffer.pull(self.batch_size)

                s_tensor = torch.from_numpy(np.array(batch_data[0])).to(
                    self.device)
                a_tensor = torch.tensor(np.array(batch_data[1]),
                                        dtype=torch.int64).unsqueeze(dim=1).to(
                    self.device)
                r_np = np.array(batch_data[2])
                ss_tensor = torch.from_numpy(np.array(batch_data[3])).to(
                    self.device)
                o_np = np.array(batch_data[4])

                q_values_np = self.now_network.forward(ss_tensor).cpu().numpy()
                max_actions = np.argmax(q_values_np, axis=1)
                q_values_np = self.delayed_network.forward(ss_tensor
                                                           ).cpu().numpy()
                q_values_lst = []
                for data_id in range(self.batch_size):
                    q_values_lst.append(
                        q_values_np[data_id][max_actions[data_id]])
                q_values_np = np.array(q_values_lst)

                target_np = r_np + o_np * self.gamma * q_values_np
                target_tensor = torch.tensor(target_np, dtype=torch.float32
                                             ).unsqueeze(dim=1).to(self.device)

            prediction = self.now_network.forward(s_tensor)
            q_values_prediction = prediction.gather(1, a_tensor)
            loss = criterion(q_values_prediction, target_tensor).to(
                self.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dnn.save_network(self.now_network, "now_network")
