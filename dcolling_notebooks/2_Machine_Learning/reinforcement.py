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
alpha = 0.2
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
