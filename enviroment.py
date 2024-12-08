import gymnasium as gym  # Import gymnasium instead of gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stock import Stock

stock_file = 'C:/Users/adird/Downloads/stock prices sp predict/sony.csv'


class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    SELL = 2
    BUY = 1
    HOLD = 0

    def __init__(self):
        super(StockTradingEnv, self).__init__()
        self.stock = Stock(stock_file, 20)
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1000, shape=(20,), dtype=np.float32)

        self.current_step = 0
        self.total_reward = 0
        self.is_in_position = False  # Tracks whether the agent is in a position (True, False)
        self.buy_price = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0
        self.buy_price = 0
        self.is_in_position = False

        initial_observation = np.array(
            self.stock.get_stock_price_until_that_index(self.current_step)
        ).astype(np.float32)

        return initial_observation, {}

    def step(self, action):
        current_price = self.stock.get_price(self.current_step)
        last_price = self.stock.get_price(self.current_step - 1)
        reward = 0
        done = True if self.stock.get_length() - 22 < self.current_step else False

        if action == self.BUY:
            if not self.is_in_position:
                self.is_in_position = True
                self.total_reward = self.total_reward + 0.01
                self.buy_price = current_price
            else:
                self.total_reward = -1  # Penalty for invalid double BUY

        elif action == self.SELL:
            if self.is_in_position:
                if current_price - self.buy_price > 0:
                    self.total_reward = 1
                self.is_in_position = False
            else:
                self.total_reward = -1  # Penalty for invalid double SELL

        elif action == self.HOLD:
            if self.is_in_position:
                self.total_reward = self.total_reward + (self.buy_price - current_price)

        else:
            raise ValueError("Invalid action")

        self.current_step += 1
        if self.current_step >= self.stock.get_length() - 2:
            done = True
        if self.total_reward == 1 or self.total_reward == -1:
            done = True

        return (
            np.array(self.stock.get_stock_price_until_that_index(self.current_step)).astype(np.float32),
            self.total_reward,
            done,
            False,
            {"total_reward": self.total_reward}
        )

    def render(self, mode='console'):

        if mode != 'console':
            raise NotImplementedError()

        #print(
        #    f"Step: {self.current_step}, Current Price: {self.stock.get_price(self.current_step - 1)}, Total Reward: {self.total_reward}")

    def close(self):
        """
        Clean up resources.
        """
        pass
