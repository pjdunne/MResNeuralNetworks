#%%

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def users_choice():
    # print("Let's play rock, paper, scissors")
    choice = random.choice(options)
    choice = "rock"
    return choice


options = ["rock", "paper", "scissors"]

document_result = []


def get_reward(user_choice, comp_choice):
    if user_choice == comp_choice:
        document_result.append("draw")
        return 0  # Draw
    elif (
        (user_choice == "rock" and comp_choice == "scissors")
        or (user_choice == "scissors" and comp_choice == "paper")
        or (user_choice == "paper" and comp_choice == "rock")
    ):
        document_result.append("win")
        return 1  # player win
    else:
        document_result.append("lose")
        return -1  # player lose


# the learning rate, Just like in supervised learning settings
alpha = 0.5
# discount rate, closer to 1 captures long-term effective reward
gamma = 0.9
# Instead of just selecting the best learned Q-value action,
# we'll sometimes favor exploring the action space further.
# Lower epsilon value results in episodes with more penalties
# (on average) which is obvious because we are exploring and
# making random decisions.
epsilon = 0.1

q_table = {}
options = ["rock", "paper", "scissors"]


def action(state, i):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(options)
    else:
        q_values = [q_table.get((state, action), 0) for action in options]
        action = options[np.argmax(q_values)]
    return action


def update_q_table(state, action, next_state, reward):
    old_q_value = q_table.get((state, action), 0)
    next_max = max(
        [q_table.get((next_state, next_action), 0) for next_action in options]
    )
    new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
    q_table[(state, action)] = new_q_value
    return q_table


def play_game(state, i):
    user_choice = users_choice()
    comp_choice = action(state, i)  # using action selected by the Q-learning algorithm

    # print("Your choice:", user_choice)
    # print("I will choose:", comp_choice)

    reward = get_reward(user_choice, comp_choice)  # calculate reward
    next_state = comp_choice  ## the next state is the current action of the computer
    ##########
    update_q_table(state, comp_choice, next_state, reward)  # update Q-table

    return reward, next_state


list_of_users_input = []
list_of_comps_input = []


def get_state(user_choice, comp_choice):
    return (user_choice, comp_choice)


contin_reward = []

num_episodes = int(1200)


def game_loop(num_episodes):
    state = random.choice(options)  # random initial state
    total_reward = 0
    for i in range(num_episodes):
        reward, state = play_game(state, i)
        total_reward += reward
        contin_reward.append(total_reward)
        list_of_users_input.append(users_choice())
        list_of_comps_input.append(state)

    # visualise reward
    average_reward = total_reward / num_episodes
    print("Average Reward over {} episodes: {}".format(num_episodes, average_reward))


game_loop(num_episodes)

### visualising results
x_var = np.linspace(0, num_episodes, num_episodes)
plt.plot(x_var, contin_reward)
plt.title("Reward Score Build-UP")
plt.xlabel("Number of Iterations")
plt.ylabel("Reward")
plt.show()


record = pd.DataFrame(
    {
        "Player's Choices": list_of_users_input,
        "Computer's Choices": list_of_comps_input,
        "Result": document_result,
    }
)

slice_result = record.iloc[-10:, :]
print(slice_result)

#%%

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def users_choice():
    # print("Let's play rock, paper, scissors")
    choice = random.choice(options)
    # choice = "rock"
    return choice


options = ["rock", "paper", "scissors"]

document_result = []


def get_reward(user_choice, comp_choice):
    if user_choice == comp_choice:
        document_result.append("draw")
        return 0  # Draw
    elif (
        (user_choice == "rock" and comp_choice == "scissors")
        or (user_choice == "scissors" and comp_choice == "paper")
        or (user_choice == "paper" and comp_choice == "rock")
    ):
        document_result.append("win")
        return 1  # player win
    else:
        document_result.append("lose")
        return -1  # player lose


# the learning rate, Just like in supervised learning settings
alpha = 0.1
# discount rate, closer to 1 captures long-term effective reward
gamma = 0.6
# Instead of just selecting the best learned Q-value action,
# we'll sometimes favor exploring the action space further.
# Lower epsilon value results in episodes with more penalties
# (on average) which is obvious because we are exploring and
# making random decisions.
epsilon = 0.1

q_table = {}
options = ["rock", "paper", "scissors"]

num_episodes = 100000
for i in range(1, num_episodes):
    epochs, reward = 0, 0
    done = False

    while not done:
        if random(0, 1) < epsilon:
            action = random.choice(options)
        else:
            action = np.argmax()


### visualising results
x_var = np.linspace(0, num_episodes, num_episodes)
plt.plot(x_var, contin_reward)
plt.title("Reward Score Build-UP")
plt.xlabel("Number of Iterations")
plt.ylabel("Reward")
plt.show()


record = pd.DataFrame(
    {
        "Player's Choices": list_of_users_input,
        "Computer's Choices": list_of_comps_input,
        "Result": document_result,
    }
)

slice_result = record.iloc[-10:, :]
print(slice_result)


# %%
