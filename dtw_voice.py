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
    page_title="Rumah Data | Dialect Intel",
    page_icon="🎙️",
    layout="wide"
)

# ============================================================
# SISTEM SETUP (FFMPEG BYPASS & AUDIO ENGINE)
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
# PROFESSIONAL NAVY THEME CSS
# ============================================================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #050a18; /* Deep Navy */
    }

    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 3rem;
        border-radius: 16px;
        margin-bottom: 2.5rem;
        border: 1px solid #1e293b;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
    }
    .main-header h1 {
        color: #38bdf8; /* Bright Cyan */
        font-weight: 800;
        letter-spacing: -1px;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Metric Cards Glassmorphism */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 3rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.7);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: 0.3s;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        background: rgba(30, 41, 59, 0.9);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 2px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    .sidebar-section {
        background: #1e293b;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-dot {
        height: 10px;
        width: 10px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px #22c55e;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 4rem 2rem 2rem;
        color: #475569;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOGIKA CORE (DIADAPTASI DARI DTW-VOICE)
# ============================================================
class AudioConfig:
    def __init__(self, k, w):
        self.SAMPLE_RATE = 16000
        self.N_MFCC = 13
        self.K_NEIGHBORS = k
        self.W_WINDOW = w

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
        # Normalisasi fitur per-audio untuk Anti-Bias Jawa
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
            cost = cost_mat[i-1, j-1]
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m] / (n + m)

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # SIDEBAR PREMIUM
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="status-indicator"><div class="status-dot"></div>CORE ENGINE ACTIVE</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Engine Parameters")
        k_val = st.slider("Neighbors Factor (K)", 1, 15, 5, help="Metrik validasi k-NN")
        dtw_w = st.slider("Time Window (W)", 20, 300, 80, step=10, help="Constraint Sakoe-Chiba (Frames)")
        st.caption("Unit: 1 Frame ≈ 10ms")
        
        st.markdown("---")
        st.markdown("### 🛠️ Maintenance")
        if st.button("Purge & Reload Database", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # HEADER
    st.markdown("""
        <div class="main-header">
            <h1>Rumah Data Dialect Fingerprinting</h1>
            <p>Sistem Analisis Pola Suara Berbasis Dynamic Time Warping & MFCC</p>
        </div>
    """, unsafe_allow_html=True)

    # DATABASE LOADING
    cfg = AudioConfig(k_val, dtw_w)
    engine = Engine(cfg)

    @st.cache_resource
    def init_db():
        templates = defaultdict(list)
        waves = defaultdict(list)
        # Menghapus listing ZIP polos sesuai permintaan user
        zip_files = list(Path('.').glob('*.zip'))
        if not zip_files: return None, None
        
        for zf in zip_files:
            cname = zf.stem.replace("Logat_", "")
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(zf, 'r') as z: z.extractall(tmp)
                for p in Path(tmp).rglob('*'):
                    if p.suffix.lower() in ['.wav', '.mp3', '.m4a']:
                        y = load_audio_safely(str(p))
                        if y is not None:
                            # Split hening tingkat tinggi
                            y_vocal = librosa.effects.split(y, top_db=20)
                            if len(y_vocal) > 0:
                                y = np.concatenate([y[s:e] for s, e in y_vocal])
                            templates[cname].append(engine.extract_features(y))
                            waves[cname].append(y)
        return templates, waves

    db_templates, db_waves = init_db()

    if db_templates is None:
        st.error("No training data detected in local directory.")
        return

    # INPUT INTERFACE
    tab1, tab2 = st.tabs(["📁 ANALYZE FILE", "🎤 CAPTURE AUDIO"])
    audio_bytes = None
    f_name = ""

    with tab1:
        file = st.file_uploader("Upload Acoustic Signal", type=['wav', 'mp3', 'm4a'])
        if file: audio_bytes, f_name = file.read(), file.name
    with tab2:
        rec = st.audio_input("Start Voice Capture")
        if rec: audio_bytes, f_name = rec.read(), "live_stream.wav"

    if audio_bytes:
        with st.spinner("Processing Signal Integrity..."):
            # Process Input
            with tempfile.NamedTemporaryFile(suffix=Path(f_name).suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name
            y_test = load_audio_safely(path)
            y_vocal_t = librosa.effects.split(y_test, top_db=20)
            if len(y_vocal_t) > 0:
                y_test = np.concatenate([y_test[s:e] for s, e in y_vocal_t])
            feat_test = engine.extract_features(y_test)
            os.remove(path)

            # DTW Calculation
            results = []
            heatmap_data = []
            heatmap_labels = []
            
            for cname, tmpls in db_templates.items():
                for idx, t in enumerate(tmpls):
                    d = dtw_dist(feat_test, t, dtw_w)
                    results.append((d, cname, idx))
                    # Simpan data untuk heatmap
                    sim_pct = (1/(1+d)*100)
                    heatmap_data.append(sim_pct)
                    heatmap_labels.append(f"{cname} #{idx+1}")

            results.sort(key=lambda x: x[0])
            best = results[0]
            
            class_min_scores = {}
            for cname in db_templates.keys():
                ds = [x[0] for x in results if x[1] == cname]
                class_min_scores[cname] = min(ds)

            # --- DISPLAY RESULTS ---
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Identitas Terdeteksi</div>
                        <div class="metric-value">{best[1].upper()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Confidence Score</div>
                        <div class="metric-value">{(1/(1+best[0])*100):.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Status Algoritma</div>
                        <div class="metric-value" style="color:#22c55e">STABLE</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # 1. HEATMAP (KEMBALI)
            st.markdown("### 🗺️ Heatmap Kemiripan Pola (Cross-Database)")
            h_matrix = np.array(heatmap_data).reshape(1, -1)
            fig_h = go.Figure(data=go.Heatmap(
                z=h_matrix, x=heatmap_labels, y=['Uji'],
                colorscale='Viridis', zmin=0, zmax=100,
                text=[[f"{v:.1f}%" for v in heatmap_data]], texttemplate="%{text}"
            ))
            fig_h.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_h, use_container_width=True)

            # 2. WAVEFORM ALIGNMENT
            st.markdown("### 📈 Temporal Signal Alignment")
            y_ref = db_waves[best[1]][best[2]]
            fig_w = make_subplots(rows=2, cols=1, vertical_spacing=0.25,
                                 subplot_titles=("Sinyal Input Test", f"Referensi Database ({best[1]})"))
            fig_w.add_trace(go.Scatter(y=y_test, line=dict(color='#38bdf8', width=1.5)), row=1, col=1)
            fig_w.add_trace(go.Scatter(y=y_ref, line=dict(color='#64748b', width=1.5)), row=2, col=1)
            fig_w.update_layout(height=500, template="plotly_dark", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_w, use_container_width=True)

            # 3. RADAR & RANKING
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown("### 🎯 Similarity Distribution Radar")
                lbls = list(class_min_scores.keys())
                vls = [1/(1+class_min_scores[l])*100 for l in lbls]
                fig_r = go.Figure(data=go.Scatterpolar(r=vls + [vls[0]], theta=lbls + [lbls[0]], fill='toself', 
                                                     fillcolor='rgba(56, 189, 248, 0.2)', line=dict(color='#38bdf8', width=3)))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_r, use_container_width=True)
            
            with c2:
                st.markdown("### 📊 Class Probability Ranking")
                ranked = sorted(class_min_scores.items(), key=lambda x: x[1])
                for name, dist in ranked:
                    sim = 1/(1+dist)*100
                    st.write(f"**{name.upper()}**")
                    st.progress(sim/100)
                    st.caption(f"Index: {sim:.2f}%")

    st.markdown('<div class="footer">Rumah Data 2026 | Built for Professional Acoustic Research</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
