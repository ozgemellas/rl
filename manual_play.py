import gymnasium as gym
import numpy as np
import pygame
from gymnasium.wrappers import RecordVideo
import os

# Configuration
ENV_ID = "LunarLanderContinuous-v3"
VIDEO_FOLDER = "videos/manual_play"
EPISODES = 3

# Action mapping logic
# LunarLander Continuous:
# Action[0]: Main engine (-1..0 off, 0..+1 throttle) ~ We map UP key to +1.0
# Action[1]: Left/Right engines (-1 right, +1 left) ~ We map LEFT/RIGHT keys
def get_action_from_keys(keys_pressed):
    main_engine = 0.0
    side_engine = 0.0
    
    # Main Engine (Up Arrow or W)
    if keys_pressed[pygame.K_UP] or keys_pressed[pygame.K_w]:
        main_engine = 1.0
    else:
        main_engine = -1.0 # Shut off
        
    # Side Engines
    if keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
        side_engine = -1.0 # Fire right engine to go left? (Depends on env specifics, usually -1 is one side)
        # In Gymnasium: -1 is fire main engine? No wait:
        # Action 0: Main engine, [-1, 1]. Range: [-1,1]. Throttle maps from 50% to 100%. 
        # Actually -1..1 maps to 0..1 for engine.
        # Action 1: Steering, [-1, 1]. -0.5..0.5 fires side engines. 
        # -1 .. -0.5 fires one, 0.5 .. 1 fires other.
        pass
        
    # Correct mapping for LunarLanderContinuous-v3
    # Action is 2 floats:
    # 0: Main engine: -1 to 1 (<=0 is off, >0 is on)
    # 1: Left/Right: -1.0..-0.5 fires left engine (pushes right), 0.5..1.0 fires right engine (pushes left)
    
    if keys_pressed[pygame.K_LEFT] or keys_pressed[pygame.K_a]:
        side_engine = 0.8 # Fire right engine (push left)
    elif keys_pressed[pygame.K_RIGHT] or keys_pressed[pygame.K_d]:
        side_engine = -0.8 # Fire left engine (push right)
    
    # For main engine, user likely wants variable control but keyboard is binary.
    # We'll give full power if pressed.
    
    return np.array([main_engine, side_engine])

def manual_play():
    # Initialize Pygame for key handling
    pygame.init()
    pygame.display.set_mode((400, 300)) # Small window to catch focus
    print("Click on the Pygame window to control the Lander.")
    print("Controls: W/Up (Engines), A/Left, D/Right")
    
    # Setup Environment
    # render_mode must be rgb_array for RecordVideo, but we also want to see it to play.
    # The 'human' render mode might mess up RecordVideo.
    # Strategy: Use rgb_array for RecVid, and manually render/blit to pygame window.
    
    env = gym.make(ENV_ID, render_mode="rgb_array")
    
    env = RecordVideo(
        env,
        video_folder=VIDEO_FOLDER,
        name_prefix="human_pilot",
        episode_trigger=lambda x: True,
        disable_logger=False
    )

    clock = pygame.time.Clock()

    for ep in range(EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        print(f"--- Episode {ep+1}/{EPISODES} Started ---")
        
        while not (done or truncated):
            # 1. Handle Events (Keyboard)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            keys = pygame.key.get_pressed()
            action = get_action_from_keys(keys)
            
            # 2. Step Environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # 3. Render for Player
            # Get the image from env (this is what RecordVideo also sees)
            frame = env.render() 
            
            # Convert to Pygame surface to show to user
            # Frame is (H, W, 3) numpy array
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            
            # Blit to display (we need to match video size roughly)
            screen = pygame.display.get_surface()
            # Resize if needed, or just blit
            # Usually env frame is 600x400
            if step == 0:
                 pygame.display.set_mode((frame.shape[1], frame.shape[0]))
                 screen = pygame.display.get_surface()
            
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            
            clock.tick(60) # Limit FPS to match gym physics (~50-60)
            step += 1
            
        print(f"Episode {ep+1} Finished. Reward: {total_reward:.2f}")

    env.close()
    pygame.quit()
    print(f"Sessions saved to {VIDEO_FOLDER}")

if __name__ == "__main__":
    manual_play()
