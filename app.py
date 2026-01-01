import streamlit as st
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from PIL import Image
import time
import os
import random

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Lunar Lander Control", 
    page_icon="🚀", 
    layout="wide"
)

# --- CSS İLE GÖRSEL İYİLEŞTİRME ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #007ACC;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        border: none;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #005A9E;
    }
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- ROKET ANİMASYONU ---
def st_rockets():
    """Ekranın altından yukarı doğru uçan roket animasyonu."""
    rockets_html = ""
    for _ in range(15):
        left_pos = random.randint(5, 95)
        duration = random.uniform(2, 5)
        delay = random.uniform(0, 1)
        size = random.randint(20, 45)
        rockets_html += f'<div style="position: fixed; bottom: -10vh; left: {left_pos}%; font-size: {size}px; z-index: 9999; animation: rise {duration}s linear {delay}s forwards; opacity: 0.8;">🚀</div>'
    
    css = """
    <style>
        @keyframes rise {
            0% { bottom: -10vh; opacity: 1; transform: rotate(0deg); }
            100% { bottom: 110vh; opacity: 0; transform: rotate(10deg); }
        }
    </style>
    """
    st.markdown(css + rockets_html, unsafe_allow_html=True)

# --- BAŞLIK ---
st.title("AI Otonom İniş Sistemi - Kontrol Paneli")
st.markdown("""
**Model:** Robust PPO (Domain Randomization ile Eğitildi)
Bu panel, otonom iniş sisteminin **farklı gök cisimleri** ve **atmosferik koşullardaki** dayanıklılığını test eder.
""")

# --- MODEL BULMA ---
possible_paths = [
    "models/PPO_Robust/ppo_lunar_robust_SUPER_5M", 
    "ppo_lunar_robust_SUPER_5M",
    "models/PPO_Robust/ppo_lunar_robust_final",
    "ppo_lunar_robust_final",
    "models/PPO/ppo_lunar_continuous_final",
    "ppo_lunar_continuous"
]
selected_model_path = None
for p in possible_paths:
    if os.path.exists(f"{p}.zip"):
        selected_model_path = p
        break

# --- SIDEBAR (PARAMETRELER) ---
st.sidebar.header("⚙️ Simülasyon Parametreleri")

st.sidebar.subheader("📍 Hedef Gök Cismi")
gravity = st.sidebar.slider("Yerçekimi (Gravity)", -11.9, -1.62, -9.8, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🌪️ Atmosferik & Bozucu Etkiler")
wind = st.sidebar.slider("Yanal Rüzgar / İtki Sapması", 0.0, 10.0, 0.0, step=0.5)
turbulence = st.sidebar.slider("Türbülans / Sensör Gürültüsü", 0.0, 2.0, 0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Zaman Kontrolü")
# Hız ayarı: Düşük değer = Hızlı, Yüksek değer = Yavaş
# Kullanıcıya "Hız" olarak sunuyoruz, ama kodda "Sleep" olarak kullanacağız.
speed_selection = st.sidebar.select_slider(
    "Animasyon Hızı",
    options=["Ağır Çekim", "Normal", "Hızlı", "Maksimum"],
    value="Normal"
)

# Hız ayarını bekleme süresine çevir
speed_map = {
    "Ağır Çekim": 0.05,
    "Normal": 0.02,
    "Hızlı": 0.005,
    "Maksimum": 0.0
}
sleep_time = speed_map[speed_selection]

# --- RISK ANALİZİ ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Görev Risk Analizi")
difficulty_score = (wind * 2.5) + (turbulence * 8) + (abs(gravity + 9.8) * 3)

if difficulty_score < 10: st.sidebar.success("✅ RİSK: DÜŞÜK (Nominal)")
elif difficulty_score < 25: st.sidebar.warning("⚠️ RİSK: ORTA (Dikkat)")
else: 
    st.sidebar.error("🚨 RİSK: YÜKSEK (Limit Zorlanıyor)")
    st.sidebar.caption("Otonom pilot tam kapasite çalışacak.")

if selected_model_path:
    st.sidebar.success(f"Sistem Hazır:\n{os.path.basename(selected_model_path)}")
else:
    st.sidebar.error("Model Bulunamadı!")

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_model(path):
    return PPO.load(path)

# --- ANA EKRAN DÜZENİ ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📺 Canlı Görüntü")
    image_placeholder = st.empty()
    start_button = st.button("SİMÜLASYONU BAŞLAT", type="primary")

with col2:
    st.subheader("📡 Canlı Telemetri")
    m1, m2 = st.columns(2)
    reward_metric = m1.empty()
    status_metric = m2.empty()
    
    st.divider()
    
    # Grafik Alanları
    st.markdown("**📉 İrtifa ve Hız Analizi**")
    chart_altitude = st.line_chart([], height=150)
    
    st.markdown("**🔥 Motor Kullanımı (Action)**")
    chart_engines = st.line_chart([], height=150)

# --- SİMÜLASYON ---
if start_button and selected_model_path:
    model = load_model(selected_model_path)
    
    try: env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array", gravity=gravity)
    except: env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        
    obs, info = env.reset()
    try: env.unwrapped.world.gravity = (0, gravity)
    except: pass

    done = False
    truncated = False
    total_reward = 0
    step = 0
    
    # --- DÖNGÜ ---
    while not (done or truncated):
        # 1. Rüzgar ve Türbülans Etkisi
        try:
            lander = env.unwrapped.lander
            wind_force = (np.random.uniform(0.9, 1.1) * wind)
            turb_x = np.random.uniform(-1, 1) * turbulence
            turb_y = np.random.uniform(-1, 1) * turbulence
            lander.ApplyForceToCenter((wind_force + turb_x, turb_y), True)
        except: pass
            
        # 2. Model Tahmini
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # 3. GÖRSEL GÜNCELLEME (Frame Skipping: 3)
        # Grafikleri ve görüntüyü her 3 adımda bir güncelle (Performans için)
        if step % 3 == 0:
            frame = env.render()
            img = Image.fromarray(frame)
            image_placeholder.image(img, use_container_width=True, caption=f"T+{step} | Hız: {speed_selection}")
            
            reward_metric.metric("Puan", f"{total_reward:.0f}")
            if total_reward > 0: status_metric.success("Stabil")
            else: status_metric.error("Kritik")
            
            new_data_telemetry = pd.DataFrame({
                "Dikey Hız": [obs[3]],
                "Yükseklik": [obs[1]]
            })
            
            new_data_engines = pd.DataFrame({
                "Ana Motor": [action[0]],
                "Yan Motorlar": [action[1]]
            })

            chart_altitude.add_rows(new_data_telemetry)
            chart_engines.add_rows(new_data_engines)
        
        # 4. HIZ KONTROLÜ (Dinamik Bekleme)
        # Bu bekleme her adımda yapılır, böylece fizik motoru yavaşlar.
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()
    
    st.markdown("---")
    if total_reward > 200:
        st_rockets()
        st.success(f"✅ GÖREV BAŞARIYLA TAMAMLANDI! (Final: {total_reward:.2f})")
    elif total_reward > 50:
        st.warning(f"⚠️ GÜVENLİ AMA SERT İNİŞ. (Final: {total_reward:.2f})")
    else:
        st.error(f"❌ GÖREV BAŞARISIZ - KAZA. (Final: {total_reward:.2f})")

elif start_button:
    st.error("Model dosyası eksik.")
else:
    image_placeholder.info("Başlatmak için butona basın.")