import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC
import os
import time

# Configuration
TIMESTEPS = 100000  # Shortened for valid benchmark demo (increase to 1M for full run)
LOG_DIR = "logs/benchmark"
MODELS_DIR = "models/benchmark"
ENV_ID = "LunarLanderContinuous-v3"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_log(algo_class, algo_name):
    print(f"\n--- Starting training for {algo_name} ---")
    
    # Create fresh environment for each algo to avoid contamination
    env = gym.make(ENV_ID)
    
    # Initialize Agent
    # Note: SAC requires MlpPolicy (or others), PPO/A2C use MlpPolicy too.
    # We use default hyperparameters to establish a "out-of-the-box" baseline.
    model = algo_class(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR
    )
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=algo_name)
    end_time = time.time()
    
    # Save
    model.save(f"{MODELS_DIR}/{algo_name}_lunar")
    
    print(f"{algo_name} training finished in {end_time - start_time:.2f} seconds.")
    env.close()

if __name__ == "__main__":
    # List of algorithms to benchmark
    # We compare On-Policy (PPO, A2C) vs Off-Policy (SAC)
    algorithms = [
        (PPO, "PPO"),
        (A2C, "A2C"),
        (SAC, "SAC")
    ]
    
    print(f"Benchmarking {len(algorithms)} algorithms on {ENV_ID} for {TIMESTEPS} steps each.")
    
    for algo_class, name in algorithms:
        train_and_log(algo_class, name)
        
    print("\nBenchmark Complete! Run the following command to visualize results:")
    print(f"tensorboard --logdir={LOG_DIR}")
