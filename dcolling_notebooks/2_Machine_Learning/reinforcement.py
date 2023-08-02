import random
import numpy as np

#### This is how a basic one to one game of rock paper and scissors work
def users_choice():
    print("Let's play rock, paper, scissors")
    while True:
        choice = input("Enter your choice (rock, paper, or scissors): ")
        if choice in ["rock", "paper", "scissors"]:
            return choice
        else:
            print("Invalid choice. Try again.")


def comp_trial():
    options = ["rock", "paper", "scissors"]
    choice = random.choice(options)
    return choice


def rule(user_choice, comp_choice):
    if user_choice == comp_choice:
        return "draw"
    elif user_choice == "rock" and comp_choice == "scissors":
        return "user wins"
    elif user_choice == "scissors" and comp_choice == "rock":
        return "computer wins"
    elif user_choice == "paper" and comp_choice == "rock":
        return "user wins"
    elif user_choice == "rock" and comp_choice == "paper":
        return "computer wins"
    elif user_choice == "scissors" and comp_choice == "paper":
        return "user wins"
    elif user_choice == "paper" and comp_choice == "scissors":
        return "computer wins"


user_choice = users_choice()
comp_choice = comp_trial()


print("Your choice:", user_choice)
print("I will choose:", comp_choice)

result = rule(user_choice, comp_choice)
print(result)

#%%-----------------------------------------------------------------------------
# tweak for reward system

import random
import numpy as np


def users_choice():
    print("Let's play rock, paper, scissors")
    while True:
        choice = input("Enter your choice (rock, paper, or scissors): ")
        if choice in ["rock", "paper", "scissors"]:
            return choice
        else:
            print("Invalid choice. Try again.")


def comp_trial():
    options = ["rock", "paper", "scissors"]
    choice = random.choice(options)
    return choice


def get_reward(user_choice, comp_choice):
    if user_choice == comp_choice:
        return 0  # Draw
    elif (
        (user_choice == "rock" and comp_choice == "scissors")
        or (user_choice == "scissors" and comp_choice == "paper")
        or (user_choice == "paper" and comp_choice == "rock")
    ):
        return 1  # Win
    else:
        return -1  # Lose


def play_game():
    user_choice = users_choice()
    comp_choice = comp_trial()

    print("Your choice:", user_choice)
    print("I will choose:", comp_choice)

    reward = get_reward(user_choice, comp_choice)
    return reward


num_episodes = 15
total_reward = 0

for episode in range(num_episodes):
    reward = play_game()
    total_reward += reward

# visualise reward, so far we are not using the reward yet
average_reward = total_reward / num_episodes
print("Average Reward over {} episodes: {}".format(num_episodes, average_reward))


#%%-----------------------------------------------------------------------------
# Q-learn Method - A method to access reinforcement learning
# let player win
# concept from - https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import random
import numpy as np


def users_choice():
    print("Let's play rock, paper, scissors")
    while True:
        choice = input("Enter your choice (rock, paper, or scissors): ")
        if choice in ["rock", "paper", "scissors"]:
            return choice
        else:
            print("Invalid choice. Try again.")


options = ["rock", "paper", "scissors"]


def comp_trial():
    choice = random.choice(options)
    return choice


def get_reward(user_choice, comp_choice):
    if user_choice == comp_choice:
        return 0  # Draw
    elif (
        (user_choice == "rock" and comp_choice == "scissors")
        or (user_choice == "scissors" and comp_choice == "paper")
        or (user_choice == "paper" and comp_choice == "rock")
    ):
        return 1  # Comp Lose
    else:
        return -1  # Comp Win


num_episodes = 20

# the learning rate, Just like in supervised learning settings
alpha = 0.5
# discount rate, closer to 1 captures long-term effective reward
gamma = 0.7
# Instead of just selecting the best learned Q-value action,
# we'll sometimes favor exploring the action space further.
# Lower epsilon value results in episodes with more penalties
# (on average) which is obvious because we are exploring and
# making random decisions.
epsilon = 0.1

q_table = {}
options = ["rock", "paper", "scissors"]


def action(state):
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


def get_state(user_choice, comp_choice):
    return (user_choice, comp_choice)


def play_game(state):
    user_choice = users_choice()
    comp_choice = action(state)  # using action selected by the Q-learning algorithm

    print("Your choice:", user_choice)
    print("I will choose:", comp_choice)

    reward = get_reward(user_choice, comp_choice)  # calculate reward
    next_state = comp_choice  # the next state is the current action of the computer

    update_q_table(state, comp_choice, next_state, reward)  # update Q-table

    return reward, next_state


def game_loop(num_episodes):
    state = random.choice(options)  # random initial state
    total_reward = 0
    for _ in range(num_episodes):
        reward, state = play_game(state)
        total_reward += reward

    # visualise reward
    average_reward = total_reward / num_episodes
    print("Average Reward over {} episodes: {}".format(num_episodes, average_reward))


game_loop(num_episodes)

#%%

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def users_choice():
    print("Let's play rock, paper, scissors")
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
        return 5  # Comp Lose
    else:
        document_result.append("lose")
        return -2  # Comp Win


num_episodes = 1000

# the learning rate, Just like in supervised learning settings
alpha = 0.9
# discount rate, closer to 1 captures long-term effective reward
gamma = 0.7
# Instead of just selecting the best learned Q-value action,
# we'll sometimes favor exploring the action space further.
# Lower epsilon value results in episodes with more penalties
# (on average) which is obvious because we are exploring and
# making random decisions.
epsilon = 0.2

q_table = {}
options = ["rock", "paper", "scissors"]


def action(state):
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


def play_game(state):
    user_choice = users_choice()
    comp_choice = action(state)  # using action selected by the Q-learning algorithm

    print("Your choice:", user_choice)
    print("I will choose:", comp_choice)

    reward = get_reward(user_choice, comp_choice)  # calculate reward
    next_state = comp_choice  # the next state is the current action of the computer

    update_q_table(state, comp_choice, next_state, reward)  # update Q-table

    return reward, next_state


list_of_users_input = []
list_of_comps_input = []


def get_state(user_choice, comp_choice):
    return (user_choice, comp_choice)


contin_reward = []


def game_loop(num_episodes):
    state = random.choice(options)  # random initial state
    total_reward = 0
    for _ in range(num_episodes):
        reward, state = play_game(state)
        total_reward += reward
        contin_reward.append(total_reward)
        list_of_users_input.append(users_choice())
        list_of_comps_input.append(play_game(state)[1])

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
    {"Player's Choices": list_of_users_input, "Computer's Choices": list_of_comps_input}
)

slice_result = record.iloc[-10:, :]
print(slice_result)

wins = document_result[document_result["result"] == "win"]
losses = document_result[document_result["result"] == "lose"]
draws = document_result[document_result["result"] == "draw"]


# %%
import pandas as pd

document_result = ["win", "lose", "draw", "win", "lose", "draw", "win", "lose", "draw"]
document_result = pd.DataFrame({"result": document_result})


wins = document_result[document_result["result"] == "win"]
losses = document_result[document_result["result"] == "lose"]
draws = document_result[document_result["result"] == "draw"]

# for i in document_result["result"]:
#     if i == "win":
#         slice = document_result[document_result["result"] == "win"]
#         wins.append(slice)

print(wins)
# %%
import pandas as pd
import numpy as np

list_of_users_input = np.array([1, 2, 5])
list_of_comps_input = np.array([3, 4, 6])


record = pd.DataFrame(
    {"Player's Choices": list_of_users_input, "Computer's Choices": list_of_comps_input}
)
print(record)
# print(record.shape[0])
slice_result = record.iloc[-10:, :]

print(slice_result)
# %%
