import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# --- CONFIGURATION ---
EXISTING_MODEL_PATH = os.path.join("models", "PPO_Robust", "ppo_lunar_robust_final")
NEW_MODEL_NAME = "ppo_lunar_robust_SUPER_5M"
MODELS_DIR_EXTENDED = os.path.join("models", "PPO_Robust_Extended")
LOG_DIR_EXTENDED = "logs_extended"
EXTRA_TIMESTEPS = 3_000_000
CHECKPOINT_FREQ = 100_000

# Ensure directories exist
os.makedirs(MODELS_DIR_EXTENDED, exist_ok=True)
os.makedirs(LOG_DIR_EXTENDED, exist_ok=True)

class DynamicEnvironmentWrapper(gym.Wrapper):
    """
    A wrapper that randomizes environment parameters at every reset:
    - Wind Power: 0.0 to 20.0
    - Turbulence Power: 0.0 to 2.0
    - Gravity: -12.0 to -1.62
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        # 1. Randomize Wind and Turbulence
        wind_power = np.random.uniform(0.0, 20.0)
        turbulence_power = np.random.uniform(0.0, 2.0)
        gravity_y = np.random.uniform(-12.0, -1.62)

        # Access the underlying environment
        unwrapped_env = self.env.unwrapped
        
        # Apply wind settings
        unwrapped_env.enable_wind = True if wind_power > 0 else False
        unwrapped_env.wind_power = wind_power
        unwrapped_env.turbulence_power = turbulence_power

        # 2. Reset the Environment (Creates new Box2D world)
        obs, info = super().reset(**kwargs)

        # 3. Randomize Gravity (Post-Reset)
        # Must be done after reset because reset recreates the world
        try:
            if hasattr(unwrapped_env, 'world') and unwrapped_env.world is not None:
                unwrapped_env.world.gravity = (0.0, gravity_y)
        except AttributeError:
            pass

        return obs, info

def make_env():
    """
    Utility function for DummyVecEnv.
    """
    env = gym.make("LunarLanderContinuous-v3")
    env = DynamicEnvironmentWrapper(env)
    return env

def main():
    print("--- STARTING EXTENDED TRAINING SESSION ---")

    # 1. Setup Vectorized Environment
    # Using DummyVecEnv for stability on Windows and to match trained model environment
    n_envs = 4
    print(f"Creating {n_envs} parallel environments (DummyVecEnv) with Dynamic Domain Randomization...")
    # DummyVecEnv accepts a list of functions that return envs
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # 2. Load Existing Model
    print(f"Loading existing model from: {EXISTING_MODEL_PATH}.zip")
    
    if not os.path.exists(EXISTING_MODEL_PATH + ".zip") and not os.path.exists(EXISTING_MODEL_PATH):
        print(f"[ERROR] Model file not found at {EXISTING_MODEL_PATH}.zip")
        print("Please ensure the file exists or check the path.")
        vec_env.close()
        return

    try:
        # Load the model and attach the new vectorized environment
        model = PPO.load(EXISTING_MODEL_PATH, env=vec_env, tensorboard_log=LOG_DIR_EXTENDED)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        vec_env.close()
        return

    # 3. Define Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // n_envs, 1), # save_freq is per env
        save_path=MODELS_DIR_EXTENDED,
        name_prefix="ppo_lunar_robust_ext"
    )

    # 4. Continue Training
    print(f"Starting training for an additional {EXTRA_TIMESTEPS} timesteps...")
    print(f"Logging to: {LOG_DIR_EXTENDED}")
    print(f"Checkpoints will be saved to: {MODELS_DIR_EXTENDED}")
    
    try:
        model.learn(
            total_timesteps=EXTRA_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="PPO_Robust_Extension",
            reset_num_timesteps=False
        )
        print("Training completed successfully.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")

    # 5. Save Final Model
    final_save_path = os.path.join("models", "PPO_Robust", NEW_MODEL_NAME)
    model.save(final_save_path)
    print(f"Final model saved to: {final_save_path}.zip")
    
    vec_env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()