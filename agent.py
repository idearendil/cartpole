from replay_buffer import Replay_Buffer
import dnn
import torch
import torch.nn as nn
import numpy as np
from exploration import boltzmann


class Cartpole_Agent():
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
        self.replay_buffer = Replay_Buffer(device=device,
                                           max_size=replay_buffer_size)

    def delayed_network_update(self):
        dnn.save_network(self.now_network, "delayed_network")
        self.delayed_network = dnn.load_network(self.device, "delayed_network")

    def __call__(self, observation):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(dim=0).to(
                self.device)
            self.now_network.eval()
            value_tensor = torch.softmax(self.now_network.forward(obs_tensor),
                                         dim=0)[0]
            return boltzmann((0, 1),
                             value_tensor.cpu().numpy(),
                             tau=self.tau)

    def train(self):

        if self.replay_buffer.size() < 5000:
            return

        self.delayed_network.eval()
        self.now_network.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.now_network.parameters(),
                                     lr=self.learning_rate)

        for batch in range(self.batch_num):

            batch_data = self.replay_buffer.pull(self.batch_size)

            s_tensor = torch.tensor(np.array(batch_data[0]),
                                    dtype=torch.float32).to(self.device)
            a_tensor = torch.tensor(np.array(batch_data[1]),
                                    dtype=torch.int64).to(self.device)
            r_np = np.array(batch_data[2])
            ss_tensor = torch.tensor(np.array(batch_data[3]),
                                     dtype=torch.float32).to(self.device)
            o_np = np.array(batch_data[4])

            q_values_np = self.now_network.forward(ss_tensor).cpu().numpy()
            max_actions = np.argmax(q_values_np, axis=1)
            q_values_np = self.delayed_network.forward(ss_tensor).cpu().numpy()
            q_values_np = np.take_along_axis(q_values_np, max_actions, axis=1)

            target_np = r_np + o_np * self.gamma * q_values_np
            target_tensor = torch.tensor(target_np,
                                         dtype=torch.float32).to(self.device)

            prediction = self.now_network.forward(s_tensor)
            q_values_prediction = prediction.gather(1, a_tensor)
            loss = criterion(q_values_prediction, target_tensor).to(
                self.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dnn.save_network(self.now_network, "now_network")
