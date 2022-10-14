#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("./game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
import torch
import torch.nn as nn

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 128 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding = 2,bias=False),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding = 1,bias=False),
                              nn.ReLU(inplace=True),
                                   ).to(torch.float64)
        self.mlp =  nn.Sequential(nn.Flatten(),
                                  nn.Linear(in_features = 6400, out_features=512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(in_features = 512, out_features=2),
                                  ).to(torch.float64)
    def forward(self,x):
        x = torch.tensor(x)
        x = x.to('cuda:2')
        x = self.conv(x)
        x = x.unsqueeze(0)
        x = self.mlp(x)
        return x

def inference(x,network):
    return network(x)

def PlayGame(args):

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        model_path = "./model.pth"
        network = torch.load(model_path)
    else:                       #We go to training mode
        network = DQN()
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
    device = 'cuda:2'
    network.to(device)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    x_t = x_t / 255.0

    s_t = np.stack((x_t, x_t, x_t, x_t)) #4*80*80
    t = 0
    while(True):
        loss = 0
        Q_sa = 0
        epoch = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = network(s_t)  # input a stack of 4 images, get the prediction
                max_Q = torch.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))


        x_t1 = x_t1 / 255.0


        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1])
        s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # 抽取小批量样本进行训练
            minibatch = random.sample(D, BATCH)
            # inputs和targets一起构成了Q值表
            # inputs = torch.zeros((BATCH, s_t.shape[0], s_t.shape[1], s_t.shape[2]))  # 32, 4, 80, 80
            targets = torch.zeros((BATCH, ACTIONS))  # 32, 2
            targets_Q = torch.zeros((BATCH, ACTIONS))  # 32, 2

            # 开始经验回放
            for i in range(0, len(minibatch)):
                # 以下序号对应D的存储顺序将信息全部取出，
                # D.append((s_t, action_index, r_t, s_t1, terminal))
                state_t = minibatch[i][0]  # 当前状态
                action_t = minibatch[i][1]  # 输入动作
                reward_t = minibatch[i][2]  # 返回奖励
                state_t1 = minibatch[i][3]  # 返回的下一状态
                terminal = minibatch[i][4]  # 返回的是否终止的标志


                # 得到预测的以输入动作x为索引的Q值列表
                targets[i] = network(state_t)
                targets_Q[i] = network(state_t)
                # 得到下一状态下预测的以输入动作x为索引的Q值列表
                Q_sa = network(state_t1)
                if terminal:  # 如果动作执行后游戏终止了，该状态下(s)该动作(a)的Q值就相当于奖励
                    targets[i, action_t] = reward_t
                    epoch += 1
                else:  # 否则，该状态(s)下该动作(a)的Q值就相当于动作执行后的即时奖励和下一状态下的最佳预期奖励乘以一个折扣率
                    targets[i, action_t] = reward_t + GAMMA * torch.max(Q_sa,axis=1)[0]

            # update network
            optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
            loss_func = torch.nn.MSELoss()
            loss_t = loss_func(targets_Q,targets)
            loss +=loss_t

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            torch.save(network,"./model.pt")

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("STATE", state,"/ epoch", epoch)

    print("Episode finished!")
    print("************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    PlayGame(args)

