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
    """
    The function `users_choice` prompts the user to enter their choice of "rock", "paper", or "scissors"
    and returns the valid choice.
    :return: the user's choice of "rock", "paper", or "scissors".
    """
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
    """
    The function "get_reward" determines the outcome of a rock-paper-scissors game between a user and a
    computer, and returns a reward value based on the result.

    :param user_choice: The choice made by the user. It can be "rock", "paper", or "scissors"
    :param comp_choice: The choice made by the computer. It can be "rock", "paper", or "scissors"
    :return: The function `get_reward` returns an integer value. It returns 0 if the user_choice and
    comp_choice are the same, indicating a draw. It returns 1 if the user_choice beats the comp_choice,
    indicating that the computer loses. It returns -1 if the comp_choice beats the user_choice,
    indicating that the computer wins.
    """
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
    """
    The function selects an action based on the epsilon-greedy policy using a Q-table.

    :param state: The "state" parameter represents the current state of the system or environment in
    which the agent is operating. It could be any relevant information or data that describes the
    current situation or context
    :return: The action that will be taken based on the current state.
    """
    if random.uniform(0, 1) < epsilon:
        action = random.choice(options)
    else:
        q_values = [q_table.get((state, action), 0) for action in options]
        action = options[np.argmax(q_values)]
    return action


def update_q_table(state, action, next_state, reward):
    """
    The function updates the Q-table based on the current state, action, next state, and reward using
    the Q-learning algorithm.

    :param state: The current state of the environment
    :param action: The action parameter represents the action taken in the current state. It is used as
    a key in the q_table dictionary to retrieve the corresponding Q-value
    :param next_state: The next_state parameter represents the state that the agent transitions to after
    taking the specified action in the current state
    :param reward: The reward is the immediate reward received after taking the action in the current
    state. It represents the feedback or evaluation of the action taken
    """
    old_q_value = q_table.get((state, action), 0)
    next_max = max(
        [q_table.get((next_state, next_action), 0) for next_action in options]
    )

    new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
    q_table[(state, action)] = new_q_value


def get_state(user_choice, comp_choice):
    return (user_choice, comp_choice)


def play_game(state):
    """
    The function "play_game" allows the user to play a game against a computer opponent using the
    Q-learning algorithm.

    :param state: The state parameter represents the current state of the game. It could be any
    information that describes the current situation or configuration of the game
    :return: the reward and the next state.
    """
    user_choice = users_choice()
    comp_choice = action(state)  # using action selected by the Q-learning algorithm

    print("Your choice:", user_choice)
    print("I will choose:", comp_choice)

    reward = get_reward(user_choice, comp_choice)  # calculate reward
    next_state = comp_choice  # the next state is the current action of the computer

    update_q_table(state, comp_choice, next_state, reward)  # update Q-table

    return reward, next_state


def game_loop(num_episodes):
    """
    The function `game_loop` runs a specified number of episodes of a game, accumulating the rewards and
    calculating the average reward.

    :param num_episodes: The parameter "num_episodes" represents the number of episodes or games that
    will be played in the game loop. Each episode consists of playing a game and updating the state and
    reward
    """
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
