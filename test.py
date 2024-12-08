print("basfjbjsa")
print("basfjbjsa")
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")

model = PPO(MlpPolicy, env, verbose=0)
print("bla")


def evaluate(
        model: BaseAlgorithm,
        num_episodes: int = 100,
        deterministic: bool = True,
) -> float:
    # This function will only work for a single environment
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")

    return mean_episode_reward


evaluate(model, num_episodes=100, deterministic=True)


