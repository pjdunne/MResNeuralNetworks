import numpy as np
import pandas as pd
import random
import pickle


class QTable:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.q_table = {
            (state, action): 0 for state in self.states for action in self.actions
        }

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def update_qvalue(self, state, action, reward, alpha, gamma):
        old_q_value = self.get_q_value(state, action)
        next_max = max(
            [self.get_q_value(state, next_action) for next_action in self.actions]
        )
        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
        self.q_table[(state, action)] = new_q_value


#### Hyperparameters###
num_episodes = 1e5
alpha = 0.3  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.5  # exploration rate

e_decay_rate = 1 / num_episodes


class Board:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.isend = False
        self.board = self.createBoard()

    def createBoard(self):
        rows = ["1", "2", "3"]
        columns = ["a", "b", "c"]
        col = {col: ["_" for _ in rows] for col in columns}
        boardgrid = pd.DataFrame(col, index=rows)
        return boardgrid

    def winner(self, move):
        board = self.board
        for _, row in board.iterrows():
            if all(val == move for val in row):
                self.isend == True
                return True

        for col in board.columns:
            if all(val == move for val in board[col]):
                self.isend == True
                return True

        if all(board.iloc[i, i] == move for i in range(3)) or all(
            board.iloc[i, 2 - i] == move for i in range(3)
        ):
            self.isend == True
            return True

        else:
            self.isend == False
            return False

    def move(self, p1, p2, epsilon):
        board = self.board
        rows = ["1", "2", "3"]
        columns = ["a", "b", "c"]
        while not self.isend:
            if random.uniform(0, 1) < epsilon:
                row = random.choice(rows)
                col = random.choice(columns)
                board.loc[row, col] = p1
            else:
                ####qtable stuff
                pass
            # pass


game = Board("p1", "p2")

# Simulate a winning condition
game.board.loc["1", "a"] = "p1"
game.board.loc["2", "a"] = "p1"
game.board.loc["3", "a"] = "p1"

if game.winner("p1"):
    print("Player 1 wins!")
else:
    print("No winner yet.")


# class State:
#     def __init__(self, rows, columns):
#         self.rows = rows
#         self.columns = columns
#         self.state = [(row, column) for row in self.rows for column in self.columns]

#         def get_state(self):
#             return self.state

#         def update_state(self, state):
#             self.state = state


# state = (rows, columns)
# action = (rows, columns)
# q_table = QTable(state, action)

# print(q_table.q_table)


# def user_choice():
#     # Ask the user where they want to play
#     return state


# def comp_choice(state, epsilon):
#     # Implement the epsilon-greedy strategy for selecting an action
#     return action


# def get_state(user_choice, comp_choice):
#     return (user_choice, comp_choice)


# def steps(action, state):
#     state = user_choice()
#     action = comp_choice(state, epsilon)
#     # What are the possible steps from the current state
#     # e.g. if I play in the top left corner, what are the possible next moves?
#     return status


# def result(action, state, status):
#     # Check if the action is a winning move
#     # if user/comp wins makes a line across the grid, return 1
#     # if no one lines are made, return 0
