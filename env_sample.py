"""
Example file of executing cartpole environment
for who doesn't know how to execute the env and
how information is given from the env.
"""


if __name__ == "__main__":

    import gym
    env = gym.make('CartPole-v1', render_mode="human")

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, _, info = env.step(action)
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
    env.close()
