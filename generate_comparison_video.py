import gymnasium as gym
import imageio
import numpy as np
import cv2
import os
from stable_baselines3 import PPO

# --- AYARLAR ---
PART1_VIDEO_PATH = r"C:\Users\90506\Desktop\python\pekistirmeli_ogrenme\Basarisiz_Model_Ruzgar_Tests.mp4"
PART2_VIDEO_PATH = r"C:\Users\90506\Desktop\python\pekistirmeli_ogrenme\videos\PPO_lunar-episode-0.mp4"
OUTPUT_FILENAME = "Karsilastirma_Heavy_Gravity.mp4"

VIDEO_FPS = 30

# --- RENKLER ---
COLOR_RED = (0, 0, 255)      # Eski Model / Başarısız
COLOR_GREEN = (0, 200, 0)    # Yeni Model / Başarılı
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

def add_text_overlay(frame, text_lines, color=COLOR_WHITE):
    """Videoya şık bir bilgi kutusu ve yazı ekler."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    
    h, w, _ = frame.shape
    # Üstte siyah bant
    cv2.rectangle(frame, (0, 0), (w, 80), COLOR_BLACK, -1)
    
    y_pos = 30
    for line in text_lines:
        cv2.putText(frame, line, (20, y_pos), font, scale, color, thickness, cv2.LINE_AA)
        y_pos += 25
    return frame

def load_video_frames(video_path):
    """Mevcut bir video dosyasından kareleri okur."""
    print(f"Mevcut video okunuyor: {video_path}")
    if not os.path.exists(video_path):
        print(f"HATA: Dosya bulunamadı -> {video_path}")
        return []

    reader = imageio.get_reader(video_path, 'ffmpeg')
    frames = []
    for i, im in enumerate(reader):
        # imageio RGB döndürür
        frames.append(im)
    
    print(f"Okunan kare sayısı: {len(frames)}")
    return frames

def process_frames(frames, scenario_text, model_text, color_theme):
    """Karelere yazı ekler."""
    processed = []
    for frame in frames:
        # OpenCV BGR ister
        f_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        overlay = [scenario_text, model_text]
        f_bgr = add_text_overlay(f_bgr, overlay, color_theme)
        
        # Geri RGB
        f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
        processed.append(f_rgb)
    return processed

def main():
    final_frames = []
    
    # --- BÖLÜM 1: EN KÖTÜ MODEL SONUCU ---
    part1_frames = load_video_frames(PART1_VIDEO_PATH)
    if part1_frames:
        processed_part1 = process_frames(
            part1_frames,
            "SENARYO: Heavy Gravity / Worst Model Test",
            "MODEL: WORST MODEL (FAIL)",
            COLOR_RED
        )
        final_frames.extend(processed_part1)
    
    # Araya siyah geçiş
    black_frame = np.zeros_like(part1_frames[0]) if part1_frames else np.zeros((400, 600, 3), dtype=np.uint8)
    for _ in range(15): final_frames.append(black_frame)

    # --- BÖLÜM 2: PPO LUNAR EPISODE 0 ---
    part2_frames = load_video_frames(PART2_VIDEO_PATH)
    if part2_frames:
        # Boyut kontrolü (Eski video ile uyuşmazsa resize et)
        if final_frames and part2_frames[0].shape != final_frames[0].shape:
            print("Uyarı: Boyut uyuşmazlığı, yeniden boyutlandırılıyor...")
            target_h, target_w = final_frames[0].shape[:2]
            part2_frames = [cv2.resize(f, (target_w, target_h)) for f in part2_frames]

        processed_part2 = process_frames(
            part2_frames,
            "SENARYO: Heavy Gravity / Robust Test",
            "MODEL: ROBUST MODEL (SUCCESS)",
            COLOR_GREEN
        )
        final_frames.extend(processed_part2)

    # --- KAYDET ---
    print(f"Video kaydediliyor: {OUTPUT_FILENAME}")
    imageio.mimsave(OUTPUT_FILENAME, final_frames, fps=VIDEO_FPS)
    print("Tamamlandı.")

if __name__ == "__main__":
    main()