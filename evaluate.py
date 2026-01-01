import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import os

# Path to the saved model
model_path = "ppo_lunar_continuous_final"
video_folder = "videos"
algorithm_name = "PPO"

# Check if model exists
if not os.path.exists(f"{model_path}.zip"):
    print(f"Error: Model file {model_path}.zip not found. Please train the model first.")
    exit()

# 1. Load Model
print(f"Loading model from {model_path}...")
model = PPO.load(model_path)

# 2. Environment Setup with Video Recording
# RecordVideo wrapper saves videos of episodes
# episode_trigger: lambda function to determine which episodes to record (here: all)
env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=video_folder,
    name_prefix=f"{algorithm_name}_lunar",
    episode_trigger=lambda x: True,  # Record every episode
    disable_logger=False
)

print("Starting evaluation...")

# 3. Evaluation Loop
episodes = 5
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    while not (done or truncated):
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        
        # Take action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")

env.close()
print(f"Videos saved to {video_folder}/")
