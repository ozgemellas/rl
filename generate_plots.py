import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
import os

# --- CHAOS WRAPPER ---
class ChaosLanderWrapper(gym.Wrapper):
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

def load_model(model_name="ppo_lunar_robust_SUPER_5M"):
    possible_paths = [
        os.path.join("models", "PPO_Robust", model_name),
        model_name,
        f"{model_name}.zip"
    ]
    for path in possible_paths:
        if os.path.exists(path + ".zip") or os.path.exists(path):
            print(f"Model found at: {path}")
            return PPO.load(path)
    raise FileNotFoundError(f"Model {model_name} not found in common paths.")

def generate_robustness_chart(model, num_episodes=20):
    print("Gürbüzlük Grafiği Oluşturuluyor...")
    scenarios = {
        "Normal":                {"grav": (-9.8, -9.8),   "wind": 0.0, "turb": 0.0},
        "Ay (Düşük Yerçekimi)":  {"grav": (-1.62, -1.62), "wind": 0.0, "turb": 0.0},
        "Fırtınalı":             {"grav": (-9.8, -9.8),   "wind": 15.0, "turb": 2.0},
        "Ekstrem (Ağır YÇ)":     {"grav": (-12.0, -12.0), "wind": 5.0,  "turb": 5.0}
    }
    
    results = {}
    
    for name, params in scenarios.items():
        env = gym.make("LunarLanderContinuous-v3")
        env = ChaosLanderWrapper(env, gravity_range=params['grav'], wind_power=params['wind'], turbulence_power=params['turb'])
        
        success_count = 0
        total_rewards = []
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = truncated = False
            ep_reward = 0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
            
            total_rewards.append(ep_reward)
            if ep_reward > 200: 
                success_count += 1
        
        success_rate = (success_count / num_episodes) * 100
        results[name] = success_rate
        print(f"  {name}: {success_rate:.1f}% Başarı")
        env.close()

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color=['#2ecc71', '#3498db', '#e67e22', '#e74c3c'])
    plt.ylim(0, 110)
    plt.ylabel("Başarı Oranı (%) (Puan > 200)")
    plt.title("Farklı Senaryolarda Ajan Gürbüzlüğü (Robustness)")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'%{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafik_gurbuzluk_performansi.png", dpi=300)
    print("Kaydedildi: grafik_gurbuzluk_performansi.png")
    plt.close()

def generate_trajectory_plot(model, num_episodes=15):
    print("Yörünge Grafiği Oluşturuluyor...")
    # Scenario: High Turbulence + Wind
    params = {"grav": (-9.8, -9.8), "wind": 5.0, "turb": 2.0}
    
    env = gym.make("LunarLanderContinuous-v3")
    env = ChaosLanderWrapper(env, gravity_range=params['grav'], wind_power=params['wind'], turbulence_power=params['turb'])
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.jet(np.linspace(0, 1, num_episodes))
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        traj_x = []
        traj_y = []
        
        while not (done or truncated):
            traj_x.append(obs[0])
            traj_y.append(obs[1])
            
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
        
        alpha = 0.6 if i < num_episodes-1 else 1.0
        linewidth = 1.5 if i < num_episodes-1 else 2.5
        plt.plot(traj_x, traj_y, color=colors[i], alpha=alpha, linewidth=linewidth)

    env.close()

    # Decorate plot
    plt.axvline(x=-0.2, color='k', linestyle='--', alpha=0.3, label='İniş Bölgesi Sınırları')
    plt.axvline(x=0.2, color='k', linestyle='--', alpha=0.3)
    
    rect = patches.Rectangle((-0.2, -0.05), 0.4, 0.05, linewidth=1, edgecolor='g', facecolor='g', alpha=0.3, label='Hedef (Pist)')
    plt.gca().add_patch(rect)
    
    plt.title(f"Ajan İniş Yörüngeleri ({num_episodes} Deneme)\nSenaryo: Rüzgarlı & Türbülanslı")
    plt.xlabel("Yatay Konum (Normalize)")
    plt.ylabel("Dikey Konum (Normalize)")
    plt.xlim(-1.0, 1.0)
    plt.ylim(-0.1, 1.5)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig("grafik_inis_yorungeleri.png", dpi=300)
    print("Kaydedildi: grafik_inis_yorungeleri.png")
    plt.close()

if __name__ == "__main__":
    try:
        model = load_model()
        generate_robustness_chart(model)
        generate_trajectory_plot(model)
        print("\nTüm grafikler başarıyla oluşturuldu!")
    except Exception as e:
        print(f"Hata oluştu: {e}")
