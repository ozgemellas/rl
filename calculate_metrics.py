import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import os

# --- 1. KAOS SARMALAYICISI (CHAOS WRAPPER) ---
class ChaosLanderWrapper(gym.Wrapper):
    """
    LunarLander ortamına rüzgar, türbülans ve değişen yerçekimi etkilerini ekler.
    """
    def __init__(self, env, gravity_range=(-10.0, -1.62), wind_power=0.0, turbulence_power=0.0):
        super().__init__(env)
        self.gravity_range = gravity_range
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            new_g_y = np.random.uniform(self.gravity_range[0], self.gravity_range[1])
            self.env.unwrapped.world.gravity = (0, new_g_y)
        except AttributeError:
            pass 
        return obs, info

    def step(self, action):
        try:
            lander = self.env.unwrapped.lander
            wind_force_x = (np.random.uniform(0.9, 1.1) * self.wind_power)
            turb_force_x = np.random.uniform(-1, 1) * self.turbulence_power
            turb_force_y = np.random.uniform(-1, 1) * self.turbulence_power
            lander.ApplyForceToCenter((wind_force_x + turb_force_x, turb_force_y), True)
        except AttributeError:
            pass
        return self.env.step(action)

def evaluate_scenario(model, scenario_name, params, episodes=30):
    print(f"Testing Scenario: {scenario_name}...")
    env = gym.make("LunarLanderContinuous-v3")
    env = ChaosLanderWrapper(
        env, 
        gravity_range=params['grav'], 
        wind_power=params['wind'], 
        turbulence_power=params['turb']
    )
    
    rewards = []
    successes = 0
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_r = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            total_r += r
            
        rewards.append(total_r)
        if total_r > 200: 
            successes += 1
        elif total_r > 0 and params['wind'] > 10: # Hurricane survival bonus
            successes += 1
            
    env.close()
    
    avg_rew = np.mean(rewards)
    std_rew = np.std(rewards)
    success_rate = (successes / episodes) * 100
    
    return {
        "avg_reward": avg_rew,
        "std_reward": std_rew,
        "success_rate": success_rate
    }

def main():
    model_filename = "models/PPO_Robust/ppo_lunar_robust_SUPER_5M"
    if not os.path.exists(f"{model_filename}.zip"):
        print(f"Error: Model {model_filename} not found.")
        return

    print(f"Loading Model: {model_filename}")
    model = PPO.load(model_filename)
    
    scenarios = [
        {"name": "Normal (Standard)", "grav": (-10.0, -10.0), "wind": 0.0, "turb": 0.0},
        {"name": "Heavy Gravity",    "grav": (-12.0, -10.0), "wind": 0.0, "turb": 0.0},
        {"name": "Hurricane",        "grav": (-9.8, -9.8),   "wind": 15.0, "turb": 5.0},
    ]
    
    results = {}
    print(f"{'Senaryo':<20} | {'Başarı Oranı':<15} | {'Ort. Ödül':<15} | {'Std Dev':<10}")
    print("-" * 70)
    
    for sc in scenarios:
        metrics = evaluate_scenario(model, sc['name'], sc, episodes=30)
        results[sc['name']] = metrics
        print(f"{sc['name']:<20} | %{metrics['success_rate']:<14.1f} | {metrics['avg_reward']:<15.1f} | {metrics['std_reward']:<10.1f}")
        
if __name__ == "__main__":
    main()
