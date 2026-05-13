import streamlit as st
import os
import zipfile
import tempfile
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Acoustic Dialect Fingerprinting",
    page_icon="🎙️",
    layout="wide"
)

# ============================================================
# SISTEM SETUP (FFMPEG BYPASS)
# ============================================================
@st.cache_resource(show_spinner=False)
def setup_audio_engine():
    try:
        import pydub
        import imageio_ffmpeg
        import librosa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "imageio-ffmpeg", "librosa"])
    return True

setup_audio_engine()
import librosa

# ============================================================
# PROFESSIONAL DARK THEME CSS
# ============================================================
st.markdown("""
<style>
    /* Global Background */
    .stApp { background-color: #0d1117; }
    
    /* Header Container */
    .main-header {
        background: #161b22;
        padding: 3rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-bottom: 2px solid #30363d;
        text-align: center;
    }
    .main-header h1 {
        color: #f0f6fc;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .main-header p {
        color: #8b949e;
        font-size: 1.1rem;
    }

    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
    }
    .metric-card {
        flex: 1;
        background: #161b22;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #30363d;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1.5px;
    }
    .metric-value {
        color: #58a6ff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    /* Sidebar Panel */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    .sidebar-title {
        color: #f0f6fc;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOGIKA CORE (DIADAPTASI DARI DTW-VOICE)
# ============================================================
class AudioConfig:
    SAMPLE_RATE = 16000
    N_MFCC = 13
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    N_FFT = 512

def load_audio_safely(filepath, sr=16000):
    try:
        import pydub
        import imageio_ffmpeg
        pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        audio = pydub.AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(sr).set_channels(1)
        return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    except:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        return y

class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def extract_features(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=self.cfg.SAMPLE_RATE, n_mfcc=self.cfg.N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.hstack([mfcc.T, delta.T, delta2.T])
        return (feat - np.mean(feat, axis=0)) / (np.std(feat, axis=0) + 1e-8)

def dtw_dist(s1, s2, w):
    n, m = len(s1), len(s2)
    w = max(w, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    cost_mat = cdist(s1, s2, metric='euclidean')
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            dp[i, j] = cost_mat[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m] / (n + m)

# ============================================================
# DASHBOARD MAIN
# ============================================================
def main():
    st.markdown("""
        <div class="main-header">
            <h1>Dialect Pattern Recognition</h1>
            <p>Sistem Analisis Akustik Berbasis Dynamic Time Warping</p>
        </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR KONFIGURASI ---
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Model Control</p>', unsafe_allow_html=True)
        k_val = st.slider("K-Neighbors Factor", 1, 10, 5, help="Jumlah sampel terdekat untuk validasi.")
        dtw_w = st.slider("W-Window Constraint", 20, 300, 80, step=10, help="Batasan pergeseran waktu dalam sekuens audio.")
        st.caption("Unit: Frames (1 frame ≈ 10ms)")
        
        st.markdown("---")
        st.markdown('<p class="sidebar-title">Database Status</p>', unsafe_allow_html=True)
        
        # Load ZIP logic
        zip_files = list(Path('.').glob('*.zip'))
        if zip_files:
            st.success(f"Aktif: {len(zip_files)} Dialek Terdeteksi")
            for zf in zip_files:
                st.caption(f"• {zf.name}")
        else:
            st.error("Data training (.zip) tidak ditemukan")

    cfg = AudioConfig()
    engine = Engine(cfg)

    # Cache Database
    @st.cache_resource
    def init_db():
        templates = defaultdict(list)
        waves = defaultdict(list)
        for zf in zip_files:
            cname = zf.stem.replace("Logat_", "")
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(zf, 'r') as z: z.extractall(tmp)
                for p in Path(tmp).rglob('*'):
                    if p.suffix.lower() in ['.wav', '.mp3', '.m4a']:
                        y = load_audio_safely(str(p))
                        if y is not None:
                            y_trim, _ = librosa.effects.trim(y, top_db=25)
                            templates[cname].append(engine.extract_features(y_trim))
                            waves[cname].append(y_trim)
        return templates, waves

    db_templates, db_waves = init_db()

    # --- INPUT SECTION ---
    tab1, tab2 = st.tabs(["📂 Analisis File Audio", "🎤 Perekaman Langsung"])
    audio_bytes = None
    f_name = ""

    with tab1:
        file = st.file_uploader("Unggah sinyal akustik", type=['wav', 'mp3', 'm4a'])
        if file: audio_bytes, f_name = file.read(), file.name
    with tab2:
        rec = st.audio_input("Rekam suara")
        if rec: audio_bytes, f_name = rec.read(), "record.wav"

    if audio_bytes:
        with st.spinner("Menghitung Integritas Sinyal..."):
            feat_test, y_test = engine.process_input(audio_bytes, f_name) # Internal helper
            
            # Perhitungan Jarak
            all_results = []
            for cname, tmpls in db_templates.items():
                for idx, t in enumerate(tmpls):
                    d = dtw_dist(feat_test, t, dtw_w)
                    all_results.append((d, cname, idx))
            
            all_results.sort(key=lambda x: x[0])
            best = all_results[0]
            
            # Confidence Logic
            class_scores = {}
            for cname in db_templates.keys():
                ds = [x[0] for x in all_results if x[1] == cname]
                class_scores[cname] = min(ds)

            # --- DISPLAY RESULTS ---
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Identitas Terdeteksi</div>
                        <div class="metric-value">{best[1].upper()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Pattern Match</div>
                        <div class="metric-value">{(1/(1+best[0])*100):.1f}%</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # 1. WAVEFORM COMPARISON
            st.markdown("### Temporal Signal Alignment")
            y_ref = db_waves[best[1]][best[2]]
            fig_wave = make_subplots(rows=2, cols=1, vertical_spacing=0.25,
                                    subplot_titles=("Sinyal Input (Uji)", f"Referensi Terdekat ({best[1]})"))
            fig_wave.add_trace(go.Scatter(y=y_test, line=dict(color='#58a6ff', width=1)), row=1, col=1)
            fig_wave.add_trace(go.Scatter(y=y_ref, line=dict(color='#8b949e', width=1)), row=2, col=1)
            fig_wave.update_layout(height=500, template="plotly_dark", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_wave, use_container_width=True)

            col_L, col_R = st.columns(2)
            
            # 2. RADAR CHART
            with col_L:
                st.markdown("### Similarity Radar")
                labels = list(class_scores.keys())
                values = [1/(1+class_scores[l])*100 for l in labels]
                fig_radar = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill='toself', line=dict(color='#58a6ff')))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_radar, use_container_width=True)

            # 3. SPECTROGRAM
            with col_R:
                st.markdown("### Power Spectrogram")
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y_test)), ref=np.max)
                fig_spec = go.Figure(data=go.Heatmap(z=D, colorscale='Viridis'))
                fig_spec.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_spec, use_container_width=True)

            # 4. RANKING LIST
            st.markdown("### Class Comparison Ranking")
            ranked = sorted(class_scores.items(), key=lambda x: x[1])
            for name, dist in ranked:
                sim = 1/(1+dist)*100
                st.write(f"**{name.upper()}**")
                st.progress(sim/100)

    st.markdown('<div class="footer">Acoustic Analysis System | Rumah Data 2026</div>', unsafe_allow_html=True)

# Helper for processing inside main
def process_input_helper(engine, audio_bytes, f_name):
    with tempfile.NamedTemporaryFile(suffix=Path(f_name).suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    y = load_audio_safely(path)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    os.remove(path)
    return engine.extract_features(y_trim), y_trim

# Patching function to engine object
Engine.process_input = process_input_helper

if __name__ == "__main__":
    main()
