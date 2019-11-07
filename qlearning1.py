import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95  # how important we find future actions value between (0,1)
EPISODES = 25000

SHOW_EVERY = 3000

# Exploration settings
epsilon = 1
START_EPSILON_DECAYING = 1

END_EPSILON_DECAYING = EPISODES // 2  # always divide into integer
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# Discretization
# make continous values more discrete, split them into bins
DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low)/DISCRETE_OS_SIZE


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    # we use this tuple to look up the 3 Q values for the available actions in the q-table
    return tuple(discrete_state.astype(np.int))


q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

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
        if render % SHOW_EVERY:
            env.render()

        # If simulation did not end yet after last step - update Q table
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

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(
            ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))  
    if episode % 100 == 0:
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
