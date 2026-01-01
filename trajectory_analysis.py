import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Ayarlar ---
# Model yolunu buraya giriniz. Mevcut dosya yapınıza göre bir yol seçildi.
# Eğer farklı bir model kullanmak isterseniz burayı değiştirebilirsiniz.
MODEL_PATH = "models/PPO_Robust/ppo_lunar_robust_SUPER_5M.zip" 
OUTPUT_FILENAME = "trajectory_plot.png"
SEED = 42

def get_trajectory(env, model, seed):
    """
    Verilen ortam ve model ile bir bölüm (episode) çalıştırır 
    ve (x, y) koordinatlarını döndürür.
    """
    obs, _ = env.reset(seed=seed)
    done = False
    coords = []
    
    # Başlangıç noktasını kaydet
    coords.append(obs[:2])
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        coords.append(obs[:2])
        done = terminated or truncated
        
    return np.array(coords)

def main():
    print("Model yükleniyor...")
    # Model var mı kontrol et, yoksa uyarı ver ama yine de dene (belki path yanlıştır)
    if not os.path.exists(MODEL_PATH):
        print(f"UYARI: Model dosyası bulunamadı: {MODEL_PATH}")
        print("Lütfen MODEL_PATH değişkenini geçerli bir yolla güncelleyin.")
        return

    model = PPO.load(MODEL_PATH)
    
    # Ortamı oluştur
    env = gym.make("LunarLanderContinuous-v3", render_mode=None)
    
    # --- Senaryo 1: İdeal (Rüzgarsız) ---
    print("Senaryo 1: İdeal koşullar çalıştırılıyor...")
    env.unwrapped.enable_wind = False
    trajectory_ideal = get_trajectory(env, model, SEED)
    
    # --- Senaryo 2: Zorlu (Rüzgarlı - Wind Power 15.0) ---
    print("Senaryo 2: Rüzgarlı koşullar (Güç: 15.0) çalıştırılıyor...")
    env.unwrapped.enable_wind = True
    env.unwrapped.wind_power = 15.0
    trajectory_windy = get_trajectory(env, model, SEED)
    
    env.close()
    
    # --- Görselleştirme ---
    print("Grafik oluşturuluyor...")
    plt.figure(figsize=(10, 8))
    
    # İdeal Yörünge
    plt.plot(trajectory_ideal[:, 0], trajectory_ideal[:, 1], 
             linestyle='--', color='green', label='İdeal (Rüzgarsız)')
    
    # Rüzgarlı Yörünge
    plt.plot(trajectory_windy[:, 0], trajectory_windy[:, 1], 
             linestyle='-', color='blue', label='Rüzgarlı (Güç: 15.0)')
    
    # İniş Alanı (Landing Pad)
    # LunarLander'da iniş alanı y=0'dadır, x ekseni -0.2 ile 0.2 arası "hedef" olarak kabul edilir ama 
    # görsel olarak landing pad genellikle biraz daha geniştir. İstenilen şekilde çiziyoruz.
    plt.plot([-0.2, 0.2], [0, 0], 'k-', linewidth=3, label='İniş Alanı')
    
    # Hedef Noktası
    plt.plot(0, 0, 'rX', markersize=10, label='Hedef Noktası')
    
    # Grafik Ayarları
    plt.title('İniş Yörüngesi Analizi: İdeal vs Rüzgarlı')
    plt.xlabel('X Koordinatı')
    plt.ylabel('Y Koordinatı')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dosyayı kaydet
    plt.savefig(OUTPUT_FILENAME)
    print(f"Sonuç kaydedildi: {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    main()
