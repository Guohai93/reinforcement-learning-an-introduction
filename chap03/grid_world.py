# -*- coding: utf-8 -*-
#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

"""
@Time    : 2018/10/8 上午9:59
@Author  : Guohai (xuguohai7@163.com)
@Declare :
    1.Codes for figures of chapter 3 in Sutton & Barto's Reinforcement Learning: An Introduction (2nd Edition)
    2.Most of codes are modified from ShangtongZhang, but rewrite the codes to make it easy to understand.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


'''global parameter'''
WORLD_ROW, WORLD_COL = 5, 5
# left, right, upper, down
ACTIONS = [
    np.array([0, -1]),
    np.array([0, +1]),
    np.array([+1, 0]),
    np.array([-1, 0])
]
ACTION_PROB = 0.25
A_POS = [0, 1]
A_NEXT_POS = [4, 1]
B_POS = [0, 3]
B_NEXT_POS = [2, 3]

DISCOUNT = 0.9


def draw_image(image):
    """
    draw 2-dim array
    """
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # add row labels
    for i, label in enumerate(range(nrows)):
        tb.add_cell(i, -1, width, height, text=label, loc='right', edgecolor='none', facecolor='none')

    # add column labels
    for j, label in enumerate(range(ncols)):
        tb.add_cell(-1, j, width, height, text=label, loc='center', edgecolor='none', facecolor='none')

    ax.add_table(tb)


def step(state, action):
    if state == A_POS:
        return A_NEXT_POS, 10
    elif state == B_POS:
        return B_NEXT_POS, 5

    next_state = (state + action).tolist()
    next_i, next_j = next_state

    if next_i < 0 or next_i >= WORLD_ROW or next_j < 0 or next_j >= WORLD_COL:
        return state, -1.0
    else:
        return [next_i, next_j], 0


def figure3_2():
    state_value = np.zeros((WORLD_ROW, WORLD_COL))
    count = 0           # count for iterations

    while True:
        # update the state_value until convergence
        new_state_value = np.zeros(state_value.shape)
        count += 1

        for i in range(WORLD_COL):
            for j in range(WORLD_ROW):
                for action in ACTIONS:
                    [next_i, next_j], reward = step([i, j], action)
                    # update the state-value based on Bellman equation
                    new_state_value[i, j] += ACTION_PROB * (reward + DISCOUNT * state_value[next_i, next_j])

        if np.sum(np.abs(state_value - new_state_value)) < 1e-4:
            draw_image(np.round(new_state_value, decimals=1))
            plt.savefig('./images/figure3_2.png')
            plt.show()
            plt.close()
            print('STA -> the number of iterations is <%d>' % count)
            break

        state_value = new_state_value


def figure3_5():
    state_value = np.zeros((WORLD_ROW, WORLD_COL))
    count = 0           # count for iterations

    while True:
        new_state_value = np.zeros(state_value.shape)
        count += 1

        for i in range(WORLD_COL):
            for j in range(WORLD_ROW):
                optimal_reward = float('-inf')
                for action in ACTIONS:
                    [next_i, next_j], immediate_reward = step([i, j], action)
                    expected_reward = immediate_reward + DISCOUNT * state_value[next_i, next_j]

                    if expected_reward >= optimal_reward:
                        optimal_i, optimal_j = next_i, next_j           # optimal policy
                        optimal_reward = expected_reward

                # update the state-value based on optimal policy
                new_state_value[i, j] = optimal_reward

        if np.sum(np.abs(state_value - new_state_value)) < 1e-4:
            draw_image(np.round(new_state_value, decimals=1))
            plt.savefig('./images/figure3_5.png')
            plt.show()
            plt.close()
            print('STA -> the number of iterations is <%d>' % count)
            break

        state_value = new_state_value


if __name__ == '__main__':
    figure3_2()

    # figure3_5()