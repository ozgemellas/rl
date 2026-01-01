import gymnasium as gym
import numpy as np
import torch as th
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration
TIMESTEPS = 2000000
MODELS_DIR = "models/PPO_Robust"
LOG_DIR = "logs/PPO_Robust"
ENV_ID = "LunarLanderContinuous-v3"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class DomainRandomizationWrapper(gym.Wrapper):
    """
    Wraps the LunarLander environment to implement Domain Randomization.
    
    Domain Randomization (DR) bridges the 'Sim-to-Real' gap by training the agent
    on a distribution of physical parameters rather than fixed values. 
    This forces the policy to be robust to dynamics variations (wind, gravity, engine power)
    instead of overfitting to a specific simulation configuration.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        # 1. Sample Randomized Parameters (The 'Uncertainty')
        # We sample uniformly from a range representing possible real-world conditions.
        
        # Wind: From 0 (Calm) to 20.0 (Hurricane/Strong Storm)
        wind_power = np.random.uniform(0.0, 20.0)
        
        # Turbulence: From 0 (Smooth) to 2.0 (Chaotic gusts)
        turbulence_power = np.random.uniform(0.0, 2.0)
        
        # Gravity: From -12.0 (Heavy planet) to -1.62 (Moon). 
        # Standard Earth is ~ -9.8.
        gravity_y = np.random.uniform(-12.0, -1.62)
        
        # 2. Inject Parameters into the Simulation
        # We access the unwrapped environment to modify internal attributes.
        env = self.env.unwrapped
        
        # Enable wind features
        env.enable_wind = True 
        env.wind_power = wind_power
        env.turbulence_power = turbulence_power
        
        # 3. Reset the Environment
        # This will create the new Box2D world and lander.
        obs, info = super().reset(**kwargs)
        
        # 4. Apply Gravity (Post-Reset)
        # Because reset() re-creates the Box2D world, we must set gravity AFTER calling strict reset.
        if hasattr(env, 'world') and env.world is not None:
            env.world.gravity = (0, gravity_y)
            
        # Optional: Print debug info for first step of occasional episodes
        # print(f"Domain Rand - Wind: {wind_power:.2f}, Grav: {gravity_y:.2f}")
            
        return obs, info

def make_env():
    """
    Utility function for multiprocessed env.
    
    :return: (gym.Env) the wrapped environment
    """
    def _init():
        env = gym.make(ENV_ID)
        env = DomainRandomizationWrapper(env)
        return env
    return _init

if __name__ == "__main__":
    print("--- Starting Robust PPO Training (Domain Randomization) ---")
    
    # 1. Vectorized Environment
    # We use 4 parallel environments to speed up data collection and diversify experience in each batch.
    # DummyVecEnv is safe for Windows. SubprocVecEnv is faster but requires careful handling.
    # We'll use DummyVecEnv to ensure maximum compatibility / stability.
    num_envs = 4
    env = DummyVecEnv([make_env() for _ in range(num_envs)])
    
    # 2. Network Architecture & Hyperparameters
    # We use a larger network [256, 256] with Tanh activation.
    # Tanh is generally preferred for continuous control tasks in PPO.
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    # 3. Model Initialization
    # Learning rate: 0.0003 (Standard robust default)
    # n_steps: 1024 * 4 envs = 4096 steps per update
    # ent_coef: 0.01 to encourage exploration in this highly variable environment
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1
    )
    
    # 4. Checkpoint Callback
    # Save the model every 100,000 steps so we don't lose progress.
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // num_envs, # Frequency is per env
        save_path=MODELS_DIR,
        name_prefix="ppo_robust"
    )
    
    # 5. Training
    print(f"Training for {TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="PPO_Robust_DomainRand"
    )
    
    # 6. Final Save
    final_path = os.path.join(MODELS_DIR, "ppo_lunar_robust_final")
    model.save(final_path)
    print(f"Training Complete. Model saved to {final_path}.zip")
    
    env.close()
