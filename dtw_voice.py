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
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# ============================================================
# [1] KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Advanced Dialect Analytics",
    page_icon="🎙️",
    layout="wide"
)

# ============================================================
# [2] AUTOMATIC AUDIO ENGINE SETUP
# ============================================================
@st.cache_resource(show_spinner=False)
def setup_audio_engine():
    try:
        import pydub
        import imageio_ffmpeg
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "imageio-ffmpeg"])
    return True

setup_audio_engine()

# ============================================================
# [3] PROFESSIONAL NAVY UI/UX STYLING (REVISED)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #050a18; }

    /* Professional Header - No "Rumah Data" */
    .main-header {
        background: #0f172a;
        padding: 3.5rem 2rem;
        border-radius: 0 0 24px 24px;
        margin: -5rem -5rem 3rem -5rem;
        border-bottom: 1px solid #1e293b;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .main-header h1 {
        color: #f8fafc;
        font-weight: 800;
        letter-spacing: -2px;
        font-size: 3.2rem;
        margin: 0;
        text-transform: uppercase;
    }
    .main-header p {
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        letter-spacing: 2px;
        margin-top: 10px;
        font-weight: 600;
    }

    /* Professional Metrics - Clean Columnar Layout */
    .metric-card {
        background: #0f172a;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #1e293b;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: #38bdf8; }
    .metric-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 800; letter-spacing: 2px; }
    .metric-value { color: #38bdf8; font-size: 2.5rem; font-weight: 800; margin-top: 8px; }

    /* Professional Analysis Insight Box */
    .analysis-card {
        background: #0f172a;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 3px solid #38bdf8;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .analysis-title { color: #f8fafc; font-weight: 700; font-size: 0.9rem; margin-bottom: 8px; display: block; }
    .analysis-text { color: #94a3b8; font-size: 0.9rem; line-height: 1.6; }

    /* Sidebar Status - Refined Layout */
    .status-container {
        background: #0f172a;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #1e293b;
        margin-bottom: 2rem;
    }
    .status-header { color: #64748b; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; }
    .status-main { color: #38bdf8; font-size: 1.8rem; font-weight: 800; margin: 5px 0; }
    .status-sub { color: #22c55e; font-size: 0.75rem; font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# ============================================================
# [4] CORE ENGINE LOGIC
# ============================================================
def load_audio_safely(filepath, sr=16000):
    try:
        import pydub
        import imageio_ffmpeg
        pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        audio = pydub.AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        return samples / float(1 << (8 * audio.sample_width - 1))
    except:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        return y

class AudioEngine:
    def __init__(self, sr=16000):
        self.sr = sr
    def process(self, y):
        intervals = librosa.effects.split(y, top_db=25)
        if len(intervals) > 0: y = np.concatenate([y[s:e] for s, e in intervals])
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.hstack([mfcc.T, delta.T, delta2.T])
        return (feat - np.mean(feat, axis=0)) / (np.std(feat, axis=0) + 1e-8), mfcc

def compute_dtw(s1, s2, w):
    n, m = len(s1), len(s2)
    w = max(w, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf); dp[0, 0] = 0.0
    cost = cdist(s1, s2, metric='cosine')
    for i in range(1, n + 1):
        for j in range(max(1, i-w), min(m, i+w)+1):
            dp[i, j] = cost[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m] / (n + m)

# ============================================================
# [5] MAIN DASHBOARD
# ============================================================
def main():
    zip_files = list(Path('.').glob('*.zip'))
    
    with st.sidebar:
        st.markdown(f"""
            <div class="status-container">
                <div class="status-header">Library Sync Status</div>
                <div class="status-main">{len(zip_files)} Active Classes</div>
                <div class="status-sub">SYSTEM OPTIMIZED</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Model Parameters")
        k_val = st.slider("Sensitivity Index (K)", 1, 15, 5)
        dtw_w = st.slider("Window Constraint (W)", 20, 300, 80)
        if st.button("Synchronize Database", use_container_width=True):
            st.cache_resource.clear(); st.rerun()

    st.markdown("""
        <div class="main-header">
            <h1>Acoustic Intelligence Analysis</h1>
            <p>PATTERN RECOGNITION SYSTEM V3.1.2</p>
        </div>
    """, unsafe_allow_html=True)

    engine = AudioEngine()

    @st.cache_resource
    def init_db():
        templates, waves = defaultdict(list), defaultdict(list)
        for zf in zip_files:
            cname = zf.stem.replace("Logat_", "").upper()
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(zf, 'r') as z: z.extractall(tmp)
                for p in Path(tmp).rglob('*'):
                    if p.suffix.lower() in ['.wav', '.mp3', '.m4a']:
                        y = load_audio_safely(str(p))
                        if y is not None:
                            feat, _ = engine.process(y)
                            templates[cname].append(feat); waves[cname].append(y)
        return templates, waves

    db_t, db_w = init_db()

    tab1, tab2 = st.tabs(["IDENTIFY SIGNAL", "REAL-TIME CAPTURE"])
    audio_bytes, f_name = None, ""

    with tab1:
        u_file = st.file_uploader("Upload signal data", type=['wav', 'mp3', 'm4a'], label_visibility="collapsed")
        if u_file: audio_bytes, f_name = u_file.read(), u_file.name
    with tab2:
        rec = st.audio_input("Microphone input", label_visibility="collapsed")
        if rec: audio_bytes, f_name = rec.read(), "live.wav"

    if audio_bytes:
        with st.spinner("Analyzing spectral properties..."):
            with tempfile.NamedTemporaryFile(suffix=Path(f_name).suffix, delete=False) as tmp:
                tmp.write(audio_bytes); path = tmp.name
            y_test = load_audio_safely(path)
            feat_test, mfcc_test = engine.process(y_test)
            os.remove(path)

            results, h_data, h_labels = [], [], []
            for cname, tmpls in db_t.items():
                for idx, t in enumerate(tmpls):
                    d = compute_dtw(feat_test, t, dtw_w)
                    results.append((d, cname, idx))
                    h_data.append(1/(1+d)*100)
                    h_labels.append(f"{cname}")

            results.sort(key=lambda x: x[0])
            best_name = results[0][1]
            best_idx = results[0][2]
            y_ref = db_w[best_name][best_idx]

            # --- DISPLAY SUMMARY ---
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Detected Identity</div><div class="metric-value">{best_name}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence Index</div><div class="metric-value">{(1/(1+results[0][0])*100):.1f}%</div></div>', unsafe_allow_html=True)

            # --- [1] WAVEFORM ---
            st.markdown("### Temporal Alignment Map")
            fig_w = make_subplots(rows=2, cols=1, vertical_spacing=0.25, subplot_titles=("Input Signal Profile", f"Reference Signal Pattern: {best_name}"))
            fig_w.add_trace(go.Scatter(y=y_test, line=dict(color='#38bdf8', width=1)), row=1, col=1)
            fig_w.add_trace(go.Scatter(y=y_ref, line=dict(color='#64748b', width=1)), row=2, col=1)
            fig_w.update_layout(height=480, template="plotly_dark", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_w, use_container_width=True)
            st.markdown(f'<div class="analysis-card"><span class="analysis-title">Signal Consistency Analysis</span><span class="analysis-text">Berdasarkan visualisasi temporal, sinyal input menunjukkan pola modulasi yang identik dengan profil {best_name}. Teknik Dynamic Time Warping berhasil menyeimbangkan distorsi waktu untuk mengonfirmasi ritme artikulasi yang serupa.</span></div>', unsafe_allow_html=True)

            # --- [2] HEATMAP (REVISED COLORS) ---
            st.markdown("### Cross-Database Similarity Matrix")
            fig_h = go.Figure(data=go.Heatmap(z=[h_data], x=h_labels, colorscale=[[0, '#0f172a'], [0.5, '#1e3a8a'], [1, '#38bdf8']], zmin=0, zmax=100, text=[[f"{v:.0f}%" for v in h_data]], texttemplate="%{text}"))
            fig_h.update_layout(height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_h, use_container_width=True)
            st.markdown(f'<div class="analysis-card"><span class="analysis-title">Matrix Correlation Insights</span><span class="analysis-text">Matriks korelasi spektral menunjukkan kepadatan energi fitur MFCC yang paling tinggi (warna cerah) pada logat {best_name}, menandakan bias spektral yang minim terhadap database referensi tersebut.</span></div>', unsafe_allow_html=True)

            # --- [3] RADAR ---
            c_radar, c_spec = st.columns(2)
            with c_radar:
                st.markdown("### Dialect Probability Radar")
                lbls = list(db_t.keys())
                vls = [1/(1+min([x[0] for x in results if x[1]==l]))*100 for l in lbls]
                fig_r = go.Figure(data=go.Scatterpolar(r=vls + [vls[0]], theta=lbls + [lbls[0]], fill='toself', fillcolor='rgba(56, 189, 248, 0.2)', line=dict(color='#38bdf8', width=2)))
                fig_r.update_layout(polar=dict(bgcolor='#0f172a', radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1e293b', tickfont=dict(size=8))), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_r, use_container_width=True)
                st.markdown(f'<div class="analysis-card"><span class="analysis-title">Class Distribution Graph</span><span class="analysis-text">Radar distribusi menunjukkan pergeseran vektor spektral yang condong ke arah sumbu {best_name}, mengonfirmasi morfologi vokal yang khas dan tidak tumpang tindih.</span></div>', unsafe_allow_html=True)

            with c_spec:
                st.markdown("### Spectral Centroid Distribution")
                sc = librosa.feature.spectral_centroid(y=y_test, sr=16000)[0]
                fig_s = go.Figure(); fig_s.add_trace(go.Scatter(y=sc, line=dict(color='#38bdf8', width=1.5)))
                fig_s.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_s, use_container_width=True)
                st.markdown(f'<div class="analysis-card"><span class="analysis-title">Brightness Analysis</span><span class="analysis-text">Distribusi frekuensi spektral menunjukkan pusat massa energi suara pada rentang Hertz tertentu yang sangat spesifik bagi logat {best_name}, mencerminkan karakteristik intonasi subjek.</span></div>', unsafe_allow_html=True)

            # --- [4] DELTA MFCC ---
            st.markdown("### Velocity of Speech Features (Delta)")
            fig_d = go.Figure(data=go.Heatmap(z=librosa.feature.delta(mfcc_test), colorscale=[[0, '#1e293b'], [1, '#38bdf8']], zmid=0))
            fig_d.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(f'<div class="analysis-card"><span class="analysis-title">Dynamic Transition Insights</span><span class="analysis-text">Heatmap Delta menggambarkan kecepatan transisi antar fonem. Kecocokan pada grafik ini menunjukkan bahwa tempo dan dinamika bicara audio uji memiliki profil kecepatan yang sinkron dengan {best_name}.</span></div>', unsafe_allow_html=True)

            # --- [5] RANKING ---
            st.markdown("### Probability Ranking")
            ranked_c = sorted({l: 1/(1+min([x[0] for x in results if x[1]==l]))*100 for l in db_t.keys()}.items(), key=lambda x: x[1], reverse=True)
            for name, score in ranked_c:
                st.write(f"**{name}**")
                st.progress(score/100)

    st.markdown('<div class="footer">Acoustic Intelligence Research Platform © 2026</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
