import gym
import numpy as np

env = gym.make("MountainCar-v0")

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low) / DISCRETE_OS_SIZE


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


episode = 3000
discrete_state = get_discrete_state(env.reset())

done = False

q_table = np.load("qtables/{}-qtable.npy".format(episode))

while not done:
    action = np.argmax(q_table[discrete_state])
    new_state, _, done, _ = env.step(action)
    new_discrete_state = get_discrete_state(new_state)
    env.render()
    discrete_state = new_discrete_state
env.close()
