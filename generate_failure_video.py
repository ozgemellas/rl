import gymnasium as gym
import imageio
import numpy as np
import cv2
import os
from stable_baselines3 import PPO

# --- CONFIGURATION ---
MODEL_PATH = "ppo_lunar_continuous_final.zip"
OUTPUT_FILENAME = "Basarisiz_Model_Ruzgar_Tests.mp4"
VIDEO_FPS = 30
MAX_STEPS = 600

# --- CHAOS WRAPPER ---
class ChaosLanderWrapper(gym.Wrapper):
    """
    LunarLander environment wrapper to add wind, turbulence and gravity changes.
    """
    def __init__(self, env, gravity_range=(-10.0, -1.62), wind_power=0.0, turbulence_power=0.0):
        super().__init__(env)
        self.gravity_range = gravity_range
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            # Set random gravity within range
            new_g_y = np.random.uniform(self.gravity_range[0], self.gravity_range[1])
            self.env.unwrapped.world.gravity = (0, new_g_y)
        except AttributeError:
            pass
        return obs, info

    def step(self, action):
        try:
            lander = self.env.unwrapped.lander
            # 1. Wind (Continuous Thrust/Force)
            # wind parameter maps to "Yanal Rüzgar / İtki Sapması"
            wind_force_x = (np.random.uniform(0.9, 1.1) * self.wind_power)
            
            # 2. Turbulence (Random Vibration)
            turb_force_x = np.random.uniform(-1, 1) * self.turbulence_power
            turb_force_y = np.random.uniform(-1, 1) * self.turbulence_power
            
            # 3. Apply Forces
            lander.ApplyForceToCenter(
                (wind_force_x + turb_force_x, turb_force_y), 
                True
            )
        except AttributeError:
            pass
        return self.env.step(action)

def add_text_overlay(frame, text_lines, color=(0, 0, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    
    y_pos = 30
    for line in text_lines:
        cv2.putText(frame, line, (20, y_pos), font, scale, color, thickness, cv2.LINE_AA)
        y_pos += 25
    return frame

def main():
    print(f"Loading Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        return

    # Create Environment
    # User requested: "10 itme sapması ve şiddetli rüzgar"
    # Mapping: wind_power=10.0 (Matches "Itme Sapması" label in app.py)
    print("Setting up environment: Wind=10.0 (Severe)")
    
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = ChaosLanderWrapper(
        env,
        gravity_range=(-1.62, -1.62), # Standard Moon Gravity to isolate wind effect
        wind_power=10.0,              # MAX Deviation/Wind
        turbulence_power=0.0
    )
    
    model = PPO.load(MODEL_PATH)
    
    obs, _ = env.reset(seed=42)
    frames = []
    
    print("Simulating...")
    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        
        frame = env.render()
        frames.append(frame)
        
        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break
    
    env.close()
    
    # Add Freeze frame at the end
    if frames:
        last = frames[-1]
        for _ in range(30):
            frames.append(last)

    # Post-process frames with text
    final_frames = []
    for frame in frames:
        f_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        overlay_text = [
            "MODEL: OLD BASELINE (FAIL)",
            "SCENARIO: SEVERE WIND (Power: 10.0)"
        ]
        f_bgr = add_text_overlay(f_bgr, overlay_text, color=(0, 0, 255)) # Red for failure
        f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
        final_frames.append(f_rgb)

    print(f"Saving video to {OUTPUT_FILENAME}")
    imageio.mimsave(OUTPUT_FILENAME, final_frames, fps=VIDEO_FPS)
    print("Done!")

if __name__ == "__main__":
    main()
