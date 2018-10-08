# -*- coding: utf-8 -*-
#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
"""
@Time    : 2018/9/14 上午9:16
@Author  : Guohai (xuguohai7@163.com)
@Declare :
    1.Codes for figures of chapter 2 in Sutton & Barto's Reinforcement Learning: An Introduction (2nd Edition)
    2.Most of codes are modified from ShangtongZhang, but rewrite the codes to make it easy to understand.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# settings for system
import sys
if sys.platform == 'linux':
    plt.switch_backend('agg')


class EpsilonGreedy(object):
    def __init__(self, k_arm=10, epsilon=0., average_sample=True, initial_value=0., true_reward=0.,
                 step_size=0.):
        self.k = k_arm                          # number of actions
        self.epsilon = epsilon
        self.average_sample = average_sample    # [average_sample] or [exponential recency_weighted average]
        self.initial_value = initial_value      # the initial value of actions
        self.true_reward = true_reward
        self.time = 0
        self.step_size = step_size                      # constant step-size parameter

    def reset(self):
        # the true reward of actions
        self.q_true = np.random.randn(self.k) + self.true_reward

        # the estimated reward of actions
        self.q_estimated = np.zeros(self.k) + self.initial_value

        # the count of actions
        self.action_count = np.zeros(self.k)

        # the optimal action whose true reward is greatest
        self.optimal_action = np.argmax(self.q_true)

    def act(self):
        if self.epsilon == 0.:
            max_estimated = np.max(self.q_estimated)
            action_candidate = [i for i, v in enumerate(self.q_estimated) if v == max_estimated]
            return np.random.choice(action_candidate)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        else:
            max_estimated = np.max(self.q_estimated)
            action_candidate = [i for i, v in enumerate(self.q_estimated) if v == max_estimated]
            return np.random.choice(action_candidate)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1

        # count the actions
        self.action_count[action] += 1

        # estimate the value of actions
        if self.average_sample is True:         # [average_sample]
            self.q_estimated[action] += 1.0 / self.action_count[action] * (reward - self.q_estimated[action])
        else:                                   # [exponential recency_weighted average]
            self.q_estimated[action] += self.step_size * (reward - self.q_estimated[action])

        return reward


class UCB(object):
    def __init__(self, k_arm=10, average_sample=True, initial_value=0., true_reward=0.,
                 step_size=0., c=2.0):
        self.k = k_arm                          # number of actions
        self.average_sample = average_sample    # [average_sample] or [exponential recency_weighted average]
        self.initial_value = initial_value      # the initial value of actions
        self.true_reward = true_reward
        self.time = 0
        self.step_size = step_size              # constant step-size parameter
        self.c = c

    def reset(self):
        # the true reward of actions
        self.q_true = np.random.randn(self.k) + self.true_reward

        # the estimated reward of actions
        self.q_estimated = np.zeros(self.k) + self.initial_value

        # the count of actions
        self.action_count = np.zeros(self.k)

        # the optimal action whose true reward is greatest
        self.optimal_action = np.argmax(self.q_true)

    def act(self):
        # select the actions according to UCB
        q_estimated_temp = [self.q_estimated[i] + self.c * np.sqrt(np.log(self.time+1) / (self.action_count[i] + 1e-5))
                            for i in range(self.k)]
        max_estimated = np.max(q_estimated_temp)
        action_candidate = [i for i, v in enumerate(q_estimated_temp) if v == max_estimated]
        return np.random.choice(action_candidate)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1

        # count the actions
        self.action_count[action] += 1

        # estimate the value of actions
        if self.average_sample is True:         # [average_sample]
            self.q_estimated[action] += 1.0 / self.action_count[action] * (reward - self.q_estimated[action])
        else:                                   # [exponential recency_weighted average]
            self.q_estimated[action] += self.step_size * (reward - self.q_estimated[action])

        return reward


class GradientBandit(object):
    def __init__(self, k_arm=10, true_reward=0., alpha=0., if_baseline=True):
        self.k = k_arm
        self.true_reward = true_reward
        self.alpha = alpha                      # learning rate
        self.if_baseline = if_baseline          # if use the baseline
        self.time = 0

    def reset(self):
        # the true reward of actions
        self.q_true = np.random.randn(self.k) + self.true_reward

        # preference for each action (un-normalized)
        self.h = np.zeros(self.k)

        # time step
        self.time = 0

        # the optimal action whose true reward is greatest
        self.optimal_action = np.argmax(self.q_true)

        # average of all the rewards  (including time t)
        self.average_reward = 0

    def act(self):
        # choose the action based on softmax
        softmax_unnormalized = np.exp(self.h)
        self.softmax = softmax_unnormalized / sum(softmax_unnormalized)

        return np.random.choice(range(self.k), p=self.softmax)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1

        self.average_reward += 1.0 / self.time * (reward - self.average_reward)

        # update the preference
        if self.if_baseline:
            baseline = self.average_reward
        else:
            baseline = 0

        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        self.h += self.alpha * (reward - baseline) * (one_hot - self.softmax)

        return reward


def simulate(bandits, runs, times):
    best_actions_count = np.zeros((len(bandits), runs, times))
    rewards = np.zeros(best_actions_count.shape)

    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs)):
            bandit.reset()
            for t in range(times):
                action = bandit.act()
                reward = bandit.step(action)

                rewards[i, r, t] = reward
                if action == bandit.optimal_action:
                    best_actions_count[i, r, t] = 1
    best_actions_count = best_actions_count.mean(axis=1)
    rewards = rewards.mean(axis=1)

    return rewards, best_actions_count


def figure2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')

    plt.savefig('./images/figure2_1.png')
    plt.show()


def figure2_2(runs=20, times=1000):
    eps_list = [0, 0.01, 0.1]
    bandits = [EpsilonGreedy(epsilon=eps) for eps in eps_list]
    rewards, best_actions_count = simulate(bandits, runs, times)

    plt.figure(figsize=(10, 20))
    colors = ['g', 'r', 'b']

    plt.subplot(2, 1, 1)
    for y, eps, c in zip(rewards, eps_list, colors):
        plt.plot(y, label='epsilon = %.2f' % (eps), color=c)
    plt.xlabel('Steps')
    plt.ylabel('Average rewards')
    plt.legend()

    plt.subplot(2, 1, 2)
    for y, eps, c in zip(best_actions_count, eps_list, colors):
        plt.plot(y, label='epsilon = %.2f' % eps, color=c)
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.legend()

    plt.savefig('./images/figure2_2.png')
    plt.show()


def figure2_3(runs=2000, times=1000):

    eps_list = [0, 0.1]
    initial_list = [5.0, 0]
    bandits = []
    for eps, initial in zip(eps_list, initial_list):
        bandit = EpsilonGreedy(epsilon=eps, average_sample=False, initial_value=initial, step_size=0.1)
        bandits.append(bandit)

    rewards, best_actions_count = simulate(bandits, runs, times)

    # plt.figure(figsize=(10, 20))
    colors = ['g', 'r']

    # plt.subplot(2, 1, 1)
    for y, eps, c in zip(rewards, eps_list, colors):
        plt.plot(y, label='epsilon = %.2f' % (eps), color=c)
    plt.xlabel('Steps')
    plt.ylabel('Average rewards')
    plt.legend()

    # plt.subplot(2, 1, 2)
    # for y, eps, c in zip(best_actions_count, eps_list, colors):
    #     plt.plot(y, label='epsilon = %.2f' % eps, color=c)
    # plt.xlabel('Steps')
    # plt.ylabel('Optimal action')
    # plt.legend()

    plt.savefig('./images/figure2_3.png')
    plt.show()


def figure2_4(runs=2000, times=1000):
    bandits = []
    bandits.append(UCB(c=2.0))
    bandits.append(EpsilonGreedy(epsilon=0.1))
    rewards, best_actions_count = simulate(bandits, runs, times)

    plt.plot(rewards[0, :], label='UCB c = 2.0', color='b')
    plt.plot(rewards[1, :], label='epsilon = 0.1', color='g')
    plt.xlabel('Steps')
    plt.ylabel('Average rewards')
    plt.legend()

    plt.savefig('./images/figure2_4.png')
    plt.show()


def figure2_5(runs=2000, times=1000):
    bandits = []
    bandits.append(GradientBandit(if_baseline=True, alpha=0.1, true_reward=4.0))
    bandits.append(GradientBandit(if_baseline=False, alpha=0.1, true_reward=4.0))
    bandits.append(GradientBandit(if_baseline=True, alpha=0.4, true_reward=4.0))
    bandits.append(GradientBandit(if_baseline=False, alpha=0.4, true_reward=4.0))

    rewards, best_actions_count = simulate(bandits, runs, times)
    label = ['with baseline, alpha = 0.1', 'without baseline, alpha = 0.1',
             'with baseline, alpha = 0.4', 'without baseline, alpha = 0.4']
    color = ['r', 'g', 'b', 'y']

    for b, l, c in zip(best_actions_count, label, color):
        plt.plot(b, label=l, color=c)
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.legend()

    plt.savefig('./images/figure2_5.png')
    plt.show()


def figure2_6(runs=2000, times=1000):
    generators = [
        lambda epsilon: EpsilonGreedy(epsilon=epsilon),
        lambda initial: EpsilonGreedy(average_sample=False, initial_value=initial, step_size=0.1),
        lambda c: UCB(c=c),
        lambda alpha: GradientBandit(alpha=alpha)
    ]
    parameters = [
        np.arange(-7, -1, dtype=np.float64),
        np.arange(-2, 3, dtype=np.float64),
        np.arange(-4, 3, dtype=np.float64),
        np.arange(-5, 2, dtype=np.float64)
    ]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for p in parameter:
            bandits.append(generator(np.power(2, p)))

    rewards, _ = simulate(bandits, runs, times)
    rewards = np.mean(rewards, axis=1)              # average over all of the steps

    label = ['epsilon-greedy', 'greedy with initialization', 'UCB', 'gradient bandit']
    color = ['r', 'k', 'b', 'g']
    i = 0
    for para, l, c in zip(parameters, label, color):
        plt.plot(para, rewards[i: i+len(para)], label=l, color=c)
        i += len(para)

    plt.xlabel('parameter')
    plt.ylabel('Average reward over first 1000 steps')
    plt.legend()

    plt.savefig('./images/figure2_6.png')
    plt.show()


if __name__ == '__main__':
    figure2_1()

    # figure2_2()

    # figure2_3()

    # figure2_4()

    # figure2_5()

    # figure2_6()


