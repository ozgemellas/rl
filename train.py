import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import torch as th

# Dizinleri oluştur
models_dir = "models/PPO"
logdir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# 1. Ortam Kurulumu (Vektörize Edilmiş)
# 4 paralel ortam oluşturuyoruz (Eğitimi hızlandırır)
def make_env():
    return gym.make("LunarLanderContinuous-v3")

# DummyVecEnv aynı işlemcide sıralı çalıştırır ama veri akışını gruplar. 
# İşlemcin çok güçlüyse SubprocVecEnv de kullanılabilir ama Windows'ta bazen sorun çıkarır.
env = DummyVecEnv([make_env for _ in range(4)]) 

# 2. Callback (Otomatik Kayıt Sistemi)
# Her 100.000 adımda bir modeli kaydeder.
checkpoint_callback = CheckpointCallback(
    save_freq=100000, 
    save_path=models_dir, 
    name_prefix="ppo_lunar"
)

# 3. Model Kurulumu (Özel Mimari ile)
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  # Tanh genelde continuous kontrol için ReLU'dan daha iyidir
    net_arch=dict(pi=[256, 256], vf=[256, 256]) # Daha derin ve geniş bir ağ (Actor ve Critic için)
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=logdir,
    learning_rate=0.0003,
    n_steps=1024,       # Paralel ortam sayısı arttığı için n_steps'i biraz düşürebiliriz
    batch_size=64,
    n_epochs=10,
    gamma=0.999,        # Gelecekteki ödüllere (iniş anına) daha fazla önem ver
    gae_lambda=0.98,
    ent_coef=0.01,      # Biraz keşfetmeye zorla (Exploration)
    policy_kwargs=policy_kwargs # Özel ağ mimarisini buraya ekledik
)

print("Starting Professional PPO training...")

# 4. Eğitim
TIMESTEPS = 1000000
model.learn(
    total_timesteps=TIMESTEPS, 
    callback=checkpoint_callback, # Callback'i buraya ekledik
    tb_log_name="PPO_Lunar_Enhanced"
)

# 5. Son Kayıt
model.save("ppo_lunar_continuous_final")
print("Training finished and model saved.")

env.close()