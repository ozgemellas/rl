import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_log(log_dir):
    print(f"Reading logs from: {log_dir}")
    
    # Find the tfevents file
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events.out.tfevents" in f]
    if not event_files:
        print("No event files found.")
        return

    # Use the most recent file if multiple exist
    event_file = max(event_files, key=os.path.getmtime)
    print(f"Processing: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    # Extract available scalar tags
    tags = ea.Tags()['scalars']
    print("Available tags:", tags)

    # We are interested in 'rollout/ep_rew_mean'
    if 'rollout/ep_rew_mean' not in tags:
        print("Tag 'rollout/ep_rew_mean' not found in logs.")
        return

    events = ea.Scalars('rollout/ep_rew_mean')
    
    steps = [e.step for e in events]
    rewards = [e.value for e in events]

    # Create a DataFrame for easier plotting/smoothing
    df = pd.DataFrame({'Timesteps': steps, 'Average Reward': rewards})

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")
    
    # Raw data (faint)
    plt.plot(df['Timesteps'], df['Average Reward'], alpha=0.3, color='blue', label='Raw Data')
    
    # Smoothed data
    df['Smoothed'] = df['Average Reward'].ewm(alpha=0.1).mean()
    plt.plot(df['Timesteps'], df['Smoothed'], color='blue', linewidth=2, label='Smoothed (EMA)')

    plt.title('Eğitim Süreci: Ödül vs. Adım Sayısı (Training Process)', fontsize=14)
    plt.xlabel('Eğitim Adımları (Timesteps)', fontsize=12)
    plt.ylabel('Ortalama Ödül (Average Reward)', fontsize=12)
    plt.legend()
    
    # Add a line for "Success" threshold (standard LunarLander is 200)
    plt.axhline(y=200, color='green', linestyle='--', label='Başarı Sınırı (200 Puan)')
    plt.legend()

    output_file = "grafik_egitim_sureci.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Grafik kaydedildi: {output_file}")
    plt.close()

if __name__ == "__main__":
    # Adjust path to match your structure
    # Based on your file checks: logs/PPO_Robust/PPO_Robust_DomainRand_1
    LOG_DIR = os.path.join("logs", "PPO_Robust", "PPO_Robust_DomainRand_1")
    
    if os.path.exists(LOG_DIR):
        plot_tensorboard_log(LOG_DIR)
    else:
        print(f"Log directory not found: {LOG_DIR}")
