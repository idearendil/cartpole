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
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
