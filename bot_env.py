

import datetime
import os
from math import floor, sqrt
import json
from pathlib import Path
import csv
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize

import pandas as pd

from gymnasium import Env, spaces

class BotEnv(Env):


    def __init__(self, config=None, df=None):
        self.df = df

        self.leverage = 100
        self.save_final_state = config['save_final_state']
        self.early_stopping = config['early_stop']
        self.act_freq = config['action_freq']
        self.init_state = config['init_state']
        self.max_steps = config['max_steps']
        self.print_rewards = config['print_rewards']
        self.save_frame = config['save_frame']
        self.s_path = config['session_path']
        self.debug = config['debug']
        
        self.instance_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S') if 'instance_id' not in config else config['instance_id']

        self.s_path.mkdir(exist_ok=True)

        self.current_step = 0
        self.reset_count = 0

        self.valid_actions = [
            0,  # hold
            1,  # buy
            2   # close all position
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        # Set this in SOME subclasses
        self.metadata = {'render.modes': ['human']}
        self.reward_range = (0, np.inf)

        # Observation space: OHLCV values, portfolio value, and more
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))

    def reset(self, seed=None):
        if self.reset_count >0:
            self.save()

        self.seed = seed
        self.current_step = 0
        self.current_cash = 300  # Starting cash
        self.current_stock = 0  # No stock held
        self.stock_price = self.df['c'].values[self.current_step]
        self.portfolio_value = self.current_cash
        self.historical_cash = [self.current_cash]
        self.historical_stock = [self.current_stock]
        self.historical_port_val = [self.portfolio_value]
        self.reset_count += 1
        self.hidden_reward = 0
        
        return self.render(), {}

    def render(self):
        obs = np.array([
            self.df['o'].values[self.current_step],
            self.df['h'].values[self.current_step],
            self.df['l'].values[self.current_step],
            self.df['c'].values[self.current_step],
            self.df['v'].values[self.current_step],
            self.current_cash,
            self.current_stock
        ])
        if self.print_rewards:
            print(f'Step: {self.current_step}, Portfolio Value: {self.historical_port_val[-1]}, Cumulated Reward: {self.hidden_reward}')

        return obs
    
    def save(self):
        with open('debug.csv', 'a') as ifile:
            writer = csv.writer(ifile, lineterminator='\n')
            writer.writerow([self.instance_id, self.reset_count, self.current_step, self.historical_port_val[-1], self.hidden_reward])
            ifile.flush()

    def step(self, action):
        # adjust policy here
        self.current_step += 1
        truncated = (self.current_step == self.max_steps) # reach timelimit

        # survived
        reward = 0

        if action == 1:  # Buy
            if self.current_stock == 0:
                stock_bought = self.current_cash/self.stock_price
                self.current_stock += stock_bought
                self.current_cash -= stock_bought * self.stock_price
                self.stock_entry = self.stock_price

        elif action == 2:  # Sell
            self.current_cash += (self.current_stock * self.stock_entry) + self.current_stock * (self.stock_price  - self.stock_entry) * self.leverage

            reward_booster = int((self.stock_price - self.stock_entry)/0.001)
            reward += reward_booster**2
            self.current_stock = 0
        
        # living expenses
        self.current_cash -=1

        self.historical_cash.append(self.current_cash)
        self.historical_stock.append(self.current_stock)
        self.historical_port_val.append(self.current_cash + (self.current_stock * self.stock_entry) + self.current_stock * (self.stock_price  - self.stock_entry) * self.leverage)
        self.stock_price = self.df['c'].values[self.current_step]

        # early exit if bankrupt
        terminated = self.historical_port_val[-1] <= 0 

        obs = self.render()        
        self.hidden_reward +=reward

        return obs, reward, terminated, truncated, {}

