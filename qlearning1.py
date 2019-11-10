import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import json

env = gym.make("MountainCar-v0")

data = {}

with open('hiperparameters.json', 'w') as fp:
    json.dump(data, fp)

# Hiperparemeters
LEARNING_RATE = data['learing_rate']
# how important we find future actions value between (0,1)
DISCOUNT = data['discount']
EPISODES = data['episodes']

SHOW_EVERY = 500

# Exploration settings
epsilon = data['epsilon']
START_EPSILON_DECAYING = data['start_epsilon_decaying']

END_EPSILON_DECAYING = data['end_epsilon_decaying']
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

save_qtable = True
render = False

# Discretization
# make continous values more discrete, split them into bins
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low) / DISCRETE_OS_SIZE


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    # we use this tuple to look up the 3 Q values for the available actions in the q-table
    return tuple(discrete_state.astype(np.int))


if save_qtable:
    folder = "/home/bt/Documents/Studia/DLR/ReinforcementLearnig/qtables"
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

q_table = np.random.uniform(low=-2,
                            high=0,
                            size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    episode_reward = 0
    # Off-policy q-learning algo learns from actions that are outside the current policy
    while not done:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        # make training faster
        if render & (episode % SHOW_EVERY == 0):
            env.render()

        # If simulation did not end update Q table
        if not done:
            # Max possible Q value in next step
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value
            current_q = q_table[discrete_state + (action, )]
            # New Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Update Q table with new Q value
            q_table[discrete_state + (action, )] = new_q
        # Restart agent
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(np.mean(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

    if episode % 10 == 0 & save_qtable:
        np.save("qtables/{}-qtable.npy".format(episode), q_table)

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.legend(loc=4)
plt.grid(True)
plt.show()
