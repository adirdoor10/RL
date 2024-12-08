import random

import gym
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_checker import check_env

from enviroment import StockTradingEnv
from stable_baselines3.common.env_util import make_vec_env


env = StockTradingEnv()
check_env(env, warn=True)
vec_env = make_vec_env(StockTradingEnv, n_envs=1)
obs, _ = env.reset()
env.render()
policy_kwargs = dict(net_arch=[128, 256,128, 64])
model = DQN(
    "MlpPolicy",
    env,
    verbose=1
).learn(2000000)
model.save("stock_trading_model")

obs, info = env.reset()
i = 0
step = 0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    #action = random.randint(0, 2)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, bla, info = env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    env.render()
    i = i+1
    step = step+1
    if done:
        print("Goal reached!", "reward=", reward)
        break


