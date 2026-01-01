import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Ayarlar ---
# Daha önce kullandığımız güçlü modeli varsayılan olarak kullanıyoruz
MODEL_PATH = "models/PPO_Robust/ppo_lunar_robust_SUPER_5M.zip" 
OUTPUT_FILENAME = "action_hist.png"
NUM_STEPS = 1000

def main():
    print("Model yükleniyor...")
    if not os.path.exists(MODEL_PATH):
        print(f"UYARI: Model dosyası bulunamadı: {MODEL_PATH}")
        print("Lütfen MODEL_PATH değişkenini geçerli bir yolla güncelleyin.")
        return

    model = PPO.load(MODEL_PATH)
    
    # Ortamı oluştur
    env = gym.make("LunarLanderContinuous-v3", render_mode=None)
    
    print(f"{NUM_STEPS} adım boyunca veri toplanıyor...")
    
    main_engine_actions = []
    
    obs, _ = env.reset(seed=42)
    
    for _ in range(NUM_STEPS):
        # Modelden aksiyon al (deterministic=True genellikle test için tercih edilir)
        action, _ = model.predict(obs, deterministic=True)
        
        # Action[0]: Ana Motor Gücü (Main Engine)
        main_engine_actions.append(action[0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    
    # --- Görselleştirme ---
    print("Histogram çiziliyor...")
    data = np.array(main_engine_actions)
    
    plt.figure(figsize=(10, 6))
    
    # Histogram: 50 bin, Mor renk
    plt.hist(data, bins=50, color='purple', alpha=0.7, edgecolor='black', label='Aksiyon Dağılımı')
    
    # Dikey Kesikli Çizgiler (-1 ve 1)
    plt.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='Motor Kapalı (-1)')
    plt.axvline(x=1, color='green', linestyle='--', linewidth=2, label='Tam Güç (1)')
    
    # Başlık ve Eksenler
    plt.title('Ana Motor İtki Gücü Dağılımı (Sürekli Kontrol Kanıtı)')
    plt.xlabel('Motor Gücü (Action[0])')
    plt.ylabel('Frekans')
    
    # Eğer veriler sadece uçlarda değilse bu notu ekle
    unique_vals = len(np.unique(data.round(decimals=2)))
    print(f"Benzersiz aksiyon değeri sayısı (yaklaşık): {unique_vals}")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Kaydet
    plt.savefig(OUTPUT_FILENAME)
    print(f"Grafik kaydedildi: {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    main()
