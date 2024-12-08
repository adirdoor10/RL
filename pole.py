import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
class GoLeftEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=10, render_mode="console"):
        super(GoLeftEnv, self).__init__()
        self.max_steps = 6
        self.render_mode = render_mode
        self.grid_size = grid_size
        self.agent_pos = random.randint(0, grid_size - 1)
        self.steps_taken = 0
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)
        self.agent_pos = random.randint(0, self.grid_size - 1)
        self.steps_taken = 0

        return np.array([self.agent_pos]).astype(np.float32), {}

    def step(self, action):
        self.steps_taken=self.steps_taken+1
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )

        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)
        reward = 1 if self.agent_pos == 5 else 0
        terminated = bool(reward)
        #reward = reward / (1 + math.log(self.steps_taken + 1))
        truncated = True if self.steps_taken >= self.max_steps else False
        if truncated:
            reward = -1

        info = {}
        return (
            np.array([self.agent_pos]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print("." * self.agent_pos, end="")
            print("x", end="")
            print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass


vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(grid_size=10))
env = GoLeftEnv(grid_size=10)
obs, _ = env.reset()
env.render()
model = A2C("MlpPolicy", env, verbose=1).learn(50000)

for time in range(10):
    obs = vec_env.reset()
    n_steps = 20
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, done, info = vec_env.step(action)
        print("obs=", obs, "reward=", reward, "done=", done)
        vec_env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break