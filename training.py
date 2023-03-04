"""
This file executes training agent from cartpole environment.
"""

import torch
import gym
from agent import CartpoleAgent


def deep_q_learning():
    """
    This function executes deep Q-learning in the cartpole environment.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    gamma = 0.98
    learning_rate = 0.003
    tau = 0.5
    batch_num = 5
    batch_size = 64
    update_frequency = 10
    replay_buffer_size = 10000
    episodes_num = 1000

    agent = CartpoleAgent(device, gamma, learning_rate,
                          tau, batch_num, batch_size,
                          replay_buffer_size, True)

    env = gym.make('CartPole-v1', render_mode="human")
    for episode in range(episodes_num):
        observation = env.reset()[0]
        for _ in range(500):
            env.render()
            action = agent(observation)
            next_observation, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push((observation, action, reward,
                                      next_observation, 1-done))
            if done:
                print(episode)
                break
            observation = next_observation
        agent.train()
        if episode % update_frequency == 0:
            agent.delayed_network_update()
            agent.tau *= 0.95


if __name__ == "__main__":
    deep_q_learning()
