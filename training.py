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

    gamma = 0.9
    learning_rate = 0.003
    tau = 0.3
    batch_num = 3
    batch_size = 128
    update_frequency = 15
    test_frequency = 50
    replay_buffer_size = 10000
    episodes_num = 1000

    agent = CartpoleAgent(device, gamma, learning_rate,
                          tau, batch_num, batch_size,
                          replay_buffer_size, True)

    # env = gym.make('CartPole-v1')
    env = gym.make('CartPole-v1', render_mode="human")
    for episode in range(episodes_num):
        observation = env.reset()[0]
        for _ in range(500):
            env.render()
            action = agent(observation)
            next_observation, reward, done, _, _ = env.step(action)
            if done:
                reward = -29
            agent.replay_buffer.push((observation, action, reward-1,
                                      next_observation, 1-done))
            if done:
                # print(episode)
                break
            observation = next_observation
        agent.train()

        if episode % update_frequency == 0:
            agent.delayed_network_update()

        if episode % test_frequency == 0:
            observation = env.reset()[0]
            for time_step in range(500):
                env.render()
                action = agent.test(observation)
                observation, reward, done, _, _ = env.step(action)
                if done:
                    print(time_step)
                    break


if __name__ == "__main__":
    deep_q_learning()
