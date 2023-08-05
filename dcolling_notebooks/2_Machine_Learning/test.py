import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

states = ["rock", "paper", "scissors"]
actions = ["rock", "paper", "scissors"]
num_episodes = int(1e4)


class QTable:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.q_table = {
            (state, action): 0 for state in self.states for action in self.actions
        }

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def update_qvalue(self, state, action, next_state, reward, alpha, gamma):
        old_q_value = self.get_q_value(state, action)
        next_max = max(
            [self.get_q_value(next_state, next_action) for next_action in self.actions]
        )
        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
        self.q_table[(state, action)] = new_q_value
        return new_q_value


q_table = QTable(states, actions)


def user_choice():
    # choice = random.choice(states)
    choice = "rock"
    return choice


def comp_choice():
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)
    else:
        state = user_choice()
        q_values = [q_table.get_q_value(state, action) for action in actions]
        action = actions[np.argmax(q_values)]
    return action


document_result = []


def reward(user_choice, comp_choice):
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


list_of_user_input = []
list_of_comp_input = []
contin_reward = []


# def game_loop(alpha, gamma, epsilon, num_episodes):
#     total_reward = 0
#     for episode in range(num_episodes):
#         # Decay the hyperparameters
#         alpha *= a_decay_rate
#         gamma *= g_decay_rate
#         epsilon *= np.exp(-e_decay_rate * episode)

#         state = user_choice()
#         action = comp_choice()

#         # visualising results
#         list_of_user_input.append(state)
#         list_of_comp_input.append(action)
#         reward_val = reward(state, action)
#         total_reward += reward_val
#         contin_reward.append(total_reward)

#         next_state = comp_choice()  # Use the existing QTable instance
#         # Use the existing QTable instance
#         q_table.update_qvalue(state, action, next_state, reward_val, alpha, gamma)
#         state = next_state
#     return contin_reward


def game_loop(alpha, gamma, epsilon, num_episodes):
    total_reward = 0
    state = user_choice()  # Initialize the state before the loop
    for episode in range(num_episodes):
        # Decay the hyperparameters
        alpha *= a_decay_rate
        gamma *= g_decay_rate
        epsilon *= np.exp(-e_decay_rate * episode)

        action = comp_choice()

        # visualising results
        list_of_user_input.append(state)
        list_of_comp_input.append(action)
        reward_val = reward(state, action)
        total_reward += reward_val
        contin_reward.append(total_reward)

        next_state = user_choice()  # Get the next state from the user's choice
        q_table.update_qvalue(state, action, next_state, reward_val, alpha, gamma)
        state = next_state  # Update the current state for the next iteration
    return contin_reward


alpha = 0.5  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.3  # exploration rate
a_decay_rate = 0.99
g_decay_rate = 0.90
e_decay_rate = 1 / num_episodes


game_loop(alpha, gamma, epsilon, num_episodes)

x_var = np.linspace(0, num_episodes, num_episodes)
plt.plot(x_var, contin_reward)
plt.title("Reward Score Build-UP")
plt.xlabel("Number of Iterations")
plt.ylabel("Reward")
# plt.show()

record = pd.DataFrame(
    {
        "Player's Choices": list_of_user_input,
        "Computer's Choices": list_of_comp_input,
        "Result": document_result,
    }
)
slice_result = record.iloc[-10:, :]
print(slice_result)
