import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
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
        
        # Not: Box2D dünyasında yerçekimini değiştirmek için world objesine erişim gerekir.
        # Gymnasium v3 sürümlerinde bu erişim bazen kısıtlıdır.
        # Erişim varsa yerçekimini değiştiriyoruz, yoksa varsayılan kalıyor.
        try:
            new_g_y = np.random.uniform(self.gravity_range[0], self.gravity_range[1])
            self.env.unwrapped.world.gravity = (0, new_g_y)
        except AttributeError:
            pass # Erişim yoksa standart yerçekimi ile devam et
            
        return obs, info

    def step(self, action):
        # --- FİZİK MÜDAHALESİ ---
        # Kuvvetleri (Rüzgar) env.step() çağırmadan ÖNCE uyguluyoruz.
        # Böylece gemi bu adımda rüzgarı hissediyor (Gecikmesiz).
        
        try:
            lander = self.env.unwrapped.lander
            
            # 1. Rüzgar (Sürekli İtme)
            # Genelde yatay eser. Hafif varyasyonla daha doğal hissettirir.
            wind_force_x = (np.random.uniform(0.9, 1.1) * self.wind_power)
            
            # 2. Türbülans (Rastgele Titreşim)
            turb_force_x = np.random.uniform(-1, 1) * self.turbulence_power
            turb_force_y = np.random.uniform(-1, 1) * self.turbulence_power
            
            # 3. Kuvveti Uygula (Kütle merkezine)
            # wake=True: Uyuyan fizik objesini uyandırır
            lander.ApplyForceToCenter(
                (wind_force_x + turb_force_x, turb_force_y), 
                True
            )
        except AttributeError:
            pass # Box2D erişimi yoksa pas geç

        # --- ADIMI GERÇEKLEŞTİR ---
        obs, reward, done, truncated, info = self.env.step(action)

        return obs, reward, done, truncated, info

# --- 2. STRES TESTİ FONKSİYONU ---
def stress_test(model_filename="ppo_lunar_robust_final", episodes=5):
    print(f"\n--- Model Stres Testi Başlatılıyor ---")
    
    # --- MODEL YÜKLEME (Düzeltilmiş Mantık) ---
    # Kodun modeli bulması için olası yolları kontrol ediyoruz
    possible_paths = [
        model_filename,                              # Direkt dosya adı
        f"models/PPO_Robust/{model_filename}",       # Robust klasörü
        f"models/PPO/{model_filename}",              # Eski klasör
        "ppo_lunar_robust_final"                     # Varsayılan ad
    ]
    
    final_path = None
    for path in possible_paths:
        # .zip uzantısı var mı diye bakıyoruz (SB3 .zip ekler)
        if os.path.exists(f"{path}.zip"):
            final_path = path
            break
            
    if final_path is None:
        print(f"HATA: Model dosyası bulunamadı! Aranan yollar: {possible_paths}")
        print("Lütfen 'train_robust.py' dosyasını çalıştırıp modeli eğittiğinden emin ol.")
        return

    print(f"Model Yükleniyor: {final_path}.zip ...")
    model = PPO.load(final_path)

    # --- SENARYOLAR ---
    scenarios = [
        {"name": "heavy_gravity", "grav": (-12.0, -10.0), "wind": 0.0, "turb": 0.0},
        {"name": "moon_storm",    "grav": (-1.62, -1.62),   "wind": 5.0, "turb": 2.0},
        {"name": "hurricane",     "grav": (-9.8, -9.8),     "wind": 15.0, "turb": 5.0},
    ]

    # --- PYGAME BAŞLATMA (Canlı İzleme İçin) ---
    pygame.init()
    screen_width, screen_height = 600, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Canlı Stres Testi - PPO Ajanı")
    clock = pygame.time.Clock()

    for sc in scenarios:
        print(f"\n>>> Senaryo Başlıyor: {sc['name']}")
        print(f"    Parametreler -> Yerçekimi: {sc['grav']}, Rüzgar: {sc['wind']}, Türbülans: {sc['turb']}")
        
        # Ortamı Oluştur
        # render_mode="rgb_array" yapıyoruz ki video kaydedebilelim.
        # Canlı izlemeyi aşağıda manuel yapacağız.
        env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        
        # Kaos Wrapper Ekle
        env = ChaosLanderWrapper(
            env, 
            gravity_range=sc['grav'], 
            wind_power=sc['wind'], 
            turbulence_power=sc['turb']
        )
        
        # Video Kaydı Ekle
        video_dir = f"videos/chaos/{sc['name']}"
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda x: True, # Tüm bölümleri kaydet
            name_prefix=f"chaos_{sc['name']}",
            disable_logger=False
        )

        successes = 0
        rewards = []

        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            total_r = 0
            
            while not (done or truncated):
                # Pygame Penceresi Kapatma Kontrolü
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        return

                # Modelden Aksiyon Al
                action, _ = model.predict(obs, deterministic=True)
                
                # Adım At
                obs, r, done, truncated, info = env.step(action)
                total_r += r
                
                # --- CANLI GÖRÜNTÜLEME (Live Rendering) ---
                frame = env.render() 
                if frame is not None:
                    # Gym Frame'ini Pygame Surface'ine çevir
                    # (Height, Width, RGB) -> (Width, Height, RGB)
                    frame = np.swapaxes(frame, 0, 1)
                    frame_surf = pygame.surfarray.make_surface(frame)
                    screen.blit(frame_surf, (0, 0))
                    pygame.display.flip()
                
                # FPS Limitleme (Çok hızlı akmasın diye)
                clock.tick(60)

            rewards.append(total_r)
            
            # Başarı Kriteri: 200 puan üstü veya pozitif puanla hayatta kalma (Fırtınada)
            status = "Çakıldı 💥"
            if total_r > 200: 
                status = "Mükemmel İniş 🚀"
                successes += 1
            elif total_r > 0:
                status = "Zorlu ama Güvenli ✅" # Fırtınada bu da başarıdır
                successes += 1 # Bunu başarı sayabiliriz veya ayrı tutabiliriz
                
            print(f"  Bölüm {ep+1}: Puan {total_r:.2f} -> {status}")

        env.close()
        success_rate = (successes / episodes) * 100
        avg_rew = np.mean(rewards)
        print(f"SENARYO SONUCU ({sc['name']}): Başarı %{success_rate:.1f} | Ort. Puan: {avg_rew:.1f}")
        print(f"Videolar kaydedildi: {video_dir}/")

    pygame.quit()
    print("\n--- Test Tamamlandı ---")

if __name__ == "__main__":
    # Eski isim: "ppo_lunar_robust_final"
    # YENİ İSİM:
    stress_test(model_filename="ppo_lunar_robust_SUPER_5M", episodes=5)