#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Bandit:
    # @kArm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @stepSize: constant step size for updating estimations
    # @sampleAverages: if True, use sample averages to update estimations instead of constant step size
    # @UCB: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradientBaseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, kArm=10, epsilon=0., initial=0., stepSize=0.1, sampleAverages=False, UCBParam=None,
                 gradient=False, gradientBaseline=False, trueReward=0.):
        self.k = kArm
        self.stepSize = stepSize
        self.sampleAverages = sampleAverages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCBParam = UCBParam
        self.gradient = gradient
        self.gradientBaseline = gradientBaseline
        self.averageReward = 0
        self.trueReward = trueReward

        # real reward for each action
        self.qTrue = []

        # estimation for each action
        self.qEst = np.zeros(self.k)

        # # of chosen times for each action
        self.actionCount = []

        self.epsilon = epsilon

        # initialize real rewards with N(0,1) distribution and estimations with desired initial value
        for i in range(0, self.k):
            self.qTrue.append(np.random.randn() + trueReward)
            self.qEst[i] = initial
            self.actionCount.append(0)

        self.bestAction = np.argmax(self.qTrue)

    # get an action for this bandit, explore or exploit?
    def getAction(self):
        # explore
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon) == 1:
                np.random.shuffle(self.indices)
                return self.indices[0]

        # exploit
        if self.UCBParam is not None:
            UCBEst = self.qEst + \
                     self.UCBParam * np.sqrt(np.log(self.time + 1) / (np.asarray(self.actionCount) + 1))
            return np.argmax(UCBEst)
        if self.gradient:
            expEst = np.exp(self.qEst)
            self.actionProb = expEst / np.sum(expEst)
            return np.random.choice(self.indices, p=self.actionProb)
        return np.argmax(self.qEst)

    # take an action, update estimation for this action
    def takeAction(self, action):
        # generate the reward under N(real reward, 1)
        self.qTrue[action] += np.random.normal(0,0.01)
        reward = np.random.randn() + self.qTrue[action]
        self.time += 1
        self.averageReward = (self.time - 1.0) / self.time * self.averageReward + reward / self.time
        self.actionCount[action] += 1

        if self.sampleAverages:
            # update estimation using sample averages
            self.qEst[action] += 1.0 / self.actionCount[action] * (reward - self.qEst[action])
        elif self.gradient:
            oneHot = np.zeros(self.k)
            oneHot[action] = 1
            if self.gradientBaseline:
                baseline = self.averageReward
            else:
                baseline = 0
            self.qEst = self.qEst + self.stepSize * (reward - baseline) * (oneHot - self.actionProb)
        else:
            # update estimation with constant step size
            self.qEst[action] += self.stepSize * (reward - self.qEst[action])
        return reward
