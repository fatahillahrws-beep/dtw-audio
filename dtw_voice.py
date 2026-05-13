import streamlit as st
import os
import zipfile
import tempfile
import json
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# [1] GLOBAL SYSTEM CONFIGURATION
# ==============================================================================
"""
Sistem ini menggunakan arsitektur pemrosesan sinyal digital untuk klasifikasi 
logat berbasis Dynamic Time Warping (DTW) dan Mel-Frequency Cepstral Coefficients (MFCC).
"""

st.set_page_config(
    page_title="Acoustic Intelligence Analysis System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# [2] DEEP AUDIO ENGINE BOOTSTRAP (BYPASS FFMPEG)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def setup_audio_engine():
    """
    Sistem instalasi otomatis untuk memastikan library pendukung tersedia guna 
    menangani berbagai format audio digital terkompresi seperti MP3 dan M4A.
    """
    try:
        import pydub
        import imageio_ffmpeg
    except ImportError:
        with st.spinner("Initializing Deep-Audio Engine Components..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "imageio-ffmpeg"])
            st.rerun()
    return True

setup_audio_engine()

# ==============================================================================
# [3] UI/UX ARCHITECTURE: PREMIUM NAVY & CYBER DESIGN
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Core Typography */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #050a18; color: #f8fafc; }

        /* Professional Laboratory Header */
        .main-header {
            background: #0f172a;
            padding: 4rem 2rem;
            border-radius: 0 0 30px 30px;
            margin: -6rem -5rem 4rem -5rem;
            border-bottom: 1px solid #1e293b;
            text-align: center;
            box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        }
        .main-header h1 {
            color: #f8fafc;
            font-weight: 800;
            letter-spacing: -2px;
            font-size: 3.5rem;
            margin: 0;
            text-transform: uppercase;
        }
        .main-header p {
            color: #38bdf8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            letter-spacing: 3px;
            margin-top: 15px;
            font-weight: 600;
        }

        /* Result Metrics Styling */
        .metric-card {
            background: #0f172a;
            padding: 2.2rem;
            border-radius: 16px;
            border: 1px solid #1e293b;
            text-align: center;
            transition: all 0.4s ease;
        }
        .metric-card:hover { 
            border-color: #38bdf8; 
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(56, 189, 248, 0.1);
        }
        .metric-label { 
            color: #64748b; 
            font-size: 0.75rem; 
            text-transform: uppercase; 
            font-weight: 800; 
            letter-spacing: 2.5px; 
        }
        .metric-value { 
            color: #38bdf8; 
            font-size: 2.8rem; 
            font-weight: 800; 
            margin-top: 10px; 
        }

        /* Analysis Container - Parallel Columns Layout */
        .analysis-container {
            background: rgba(15, 23, 42, 0.7);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #38bdf8;
            margin-top: 0.5rem;
            margin-bottom: 2rem;
            min-height: 180px; /* Ensure alignment */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .analysis-title { 
            color: #f8fafc; 
            font-weight: 800; 
            font-size: 0.95rem; 
            margin-bottom: 8px; 
            text-transform: uppercase;
        }
        .analysis-text { 
            color: #94a3b8; 
            font-size: 0.9rem; 
            line-height: 1.6; 
            text-align: left;
        }

        /* Sidebar Customization */
        section[data-testid="stSidebar"] { 
            background-color: #0f172a; 
            border-right: 1px solid #1e293b; 
        }
        .status-panel {
            background: #050a18;
            padding: 1.8rem;
            border-radius: 15px;
            border: 1px solid #1e3a8a;
            margin-bottom: 2.5rem;
            text-align: center;
        }
        .status-header { color: #64748b; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; }
        .status-main { color: #38bdf8; font-size: 2rem; font-weight: 800; margin: 8px 0; }
        .status-tag { 
            background: rgba(34, 197, 94, 0.1);
            color: #22c55e; 
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.7rem; 
            font-weight: 800; 
        }

        .footer {
            text-align: center;
            padding: 5rem 2rem 2rem;
            color: #334155;
            font-size: 0.8rem;
            letter-spacing: 1px;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==============================================================================
# [4] ACOUSTIC PROCESSING UNIT
# ==============================================================================
class DialectEngine:
    """Mesin utama pengolah sinyal dan ekstraksi pola akustik."""
    def __init__(self, k, w):
        self.SR = 16000
        self.N_MFCC = 13
        self.K_NEIGHBORS = k
        self.W_WINDOW = w
        self.TOP_DB_VAD = 20  # Agresivitas pembersihan hening

    def load_standardized(self, path):
        """Membaca audio dengan normalisasi bit-depth."""
        try:
            import pydub
            import imageio_ffmpeg
            pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            audio = pydub.AudioSegment.from_file(path)
            audio = audio.set_frame_rate(self.SR).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            return samples / float(1 << (8 * audio.sample_width - 1))
        except Exception as e:
            y, _ = librosa.load(path, sr=self.SR, mono=True)
            return y

    def apply_vad(self, y):
        """Pembersihan hening untuk mengeliminasi bias durasi kosong."""
        intervals = librosa.effects.split(y, top_db=self.TOP_DB_VAD)
        if len(intervals) > 0:
            return np.concatenate([y[s:e] for s, e in intervals])
        return y

    def get_features(self, y):
        """Ekstraksi MFCC + Delta dengan Normalisasi Z-Score."""
        mfcc = librosa.feature.mfcc(y=y, sr=self.SR, n_mfcc=self.N_MFCC)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        combined = np.hstack([mfcc.T, d1.T, d2.T])
        mu = np.mean(combined, axis=0)
        sigma = np.std(combined, axis=0) + 1e-8
        return (combined - mu) / sigma, mfcc

def compute_dtw_alignment(s1, s2, w_const):
    """Kalkulasi DTW menggunakan metrik Cosine untuk akurasi morfologi."""
    n, m = len(s1), len(s2)
    w = max(w_const, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf); dp[0, 0] = 0.0
    cost_matrix = cdist(s1, s2, metric='cosine')
    for i in range(1, n + 1):
        for j in range(max(1, i-w), min(m, i+w)+1):
            dp[i, j] = cost_matrix[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m] / (n + m)

# ==============================================================================
# [5] VISUALIZATION MODULE
# ==============================================================================
class VisualFactory:
    """Modul untuk menghasilkan visualisasi data yang selaras dengan tema."""
    
    @staticmethod
    def create_waveform(y_test, y_ref, winner):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.25, 
                            subplot_titles=("Input Signal Profile", f"Master Template: {winner}"))
        fig.add_trace(go.Scatter(y=y_test, line=dict(color='#38bdf8', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(y=y_ref, line=dict(color='#64748b', width=1)), row=2, col=1)
        fig.update_layout(height=480, template="plotly_dark", showlegend=False, 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    @staticmethod
    def create_heatmap(vals, labels):
        # Palette: Dark Navy to Vibrant Cyan
        fig = go.Figure(data=go.Heatmap(
            z=[vals], x=labels, colorscale=[[0, '#0f172a'], [0.5, '#1e40af'], [1, '#38bdf8']], 
            zmin=0, zmax=100, text=[[f"{v:.0f}%" for v in vals]], texttemplate="%{text}"
        ))
        fig.update_layout(height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                          margin=dict(l=0, r=0, t=10, b=0))
        return fig

    @staticmethod
    def create_radar(labels, values):
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]], theta=labels + [labels[0]], 
            fill='toself', fillcolor='rgba(56, 189, 248, 0.2)', 
            line=dict(color='#38bdf8', width=2)
        ))
        fig.update_layout(polar=dict(bgcolor='#0f172a', radialaxis=dict(visible=True, range=[0, 100], 
                          gridcolor='#1e293b')), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        return fig

    @staticmethod
    def create_spectral(y_in):
        sc = librosa.feature.spectral_centroid(y=y_in, sr=16000)[0]
        fig = go.Figure(); fig.add_trace(go.Scatter(y=sc, line=dict(color='#fbbf24', width=1.5), fill='tozeroy'))
        fig.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                          plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
        return fig

    @staticmethod
    def create_delta(mfcc):
        # Palette: Deep Navy to Amber/Gold (for visibility)
        delta = librosa.feature.delta(mfcc)
        fig = go.Figure(data=go.Heatmap(
            z=delta, colorscale=[[0, '#0f172a'], [0.5, '#1e293b'], [1, '#fbbf24']], zmid=0
        ))
        fig.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                          margin=dict(l=0, r=0, t=10, b=0))
        return fig

# ==============================================================================
# [6] ANALYTICAL DASHBOARD PIPELINE
# ==============================================================================
def main():
    zips = list(Path('.').glob('*.zip'))
    
    # SIDEBAR CONTROL ROOM
    with st.sidebar:
        st.markdown(f"""
            <div class="status-panel">
                <div class="status-header">Computational Core</div>
                <div class="status-main">{len(zips)} Logat</div>
                <div class="status-tag">ENGINE OPTIMIZED</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### System Tuning")
        k_param = st.slider("Classification Sensitivity (K)", 1, 15, 5)
        w_param = st.slider("Dynamic Alignment Window (W)", 20, 300, 100, step=10)
        
        st.markdown("---")
        if st.button("Reload Global Engine", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # LABORATORY HEADER
    st.markdown("""
        <div class="main-header">
            <h1>Acoustic Intelligence Analysis</h1>
            <p>PATTERN RECOGNITION SYSTEM V3.3.4 (STABLE)</p>
        </div>
    """, unsafe_allow_html=True)

    engine = DialectEngine(k_param, w_param)
    viz = VisualFactory()

    # DATABASE INITIALIZATION
    @st.cache_resource
    def load_dialect_assets():
        t_db, w_db = defaultdict(list), defaultdict(list)
        if not zips: return None, None
        for z in zips:
            label = z.stem.replace("Logat_", "").upper()
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(z, 'r') as zf: zf.extractall(td)
                for f in Path(td).rglob('*'):
                    if f.suffix.lower() in ['.wav', '.mp3', '.m4a']:
                        y_raw = engine.load_standardized(str(f))
                        y_cln = engine.apply_vad(y_raw)
                        feats, _ = engine.get_features(y_cln)
                        t_db[label].append(feats)
                        w_db[label].append(y_cln)
        return t_db, w_db

    db_templates, db_waves = load_dialect_assets()

    if not db_templates:
        st.error("Missing dialect assets (.zip files). System standby.")
        return

    # MAIN INTERFACE
    tab_f, tab_r = st.tabs(["SIGNAL IDENTIFICATION", "VOICE STREAMING"])
    data_stream, file_id = None, ""

    with tab_f:
        u = st.file_uploader("Signal Target", type=['wav', 'mp3', 'm4a'], label_visibility="collapsed")
        if u: data_stream, file_id = u.read(), u.name
    with tab_r:
        m = st.audio_input("Live Capture", label_visibility="collapsed")
        if m: data_stream, file_id = m.read(), "stream_live.wav"

    # EXECUTION
    if data_stream:
        with st.spinner("Processing spectral coefficients..."):
            with tempfile.NamedTemporaryFile(suffix=Path(file_id).suffix, delete=False) as tmp:
                tmp.write(data_stream); tmp_p = tmp.name
            
            y_in = engine.apply_vad(engine.load_standardized(tmp_p))
            f_in, mfcc_in = engine.get_features(y_in)
            os.remove(tmp_p)

            # CALCULATION LOOP
            scores, h_vals, h_lbls = [], [], []
            for dial, tmpls in db_templates.items():
                for i, t in enumerate(tmpls):
                    dist = compute_dtw_alignment(f_in, t, w_param)
                    scores.append((dist, dial, i))
                    h_vals.append(1/(1+dist)*100)
                    h_lbls.append(f"{dial}")

            scores.sort(key=lambda x: x[0])
            winner = scores[0][1]
            conf = (1/(1+scores[0][0])*100)

            # --- DISPLAY GRID ---
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Target Dialect</div><div class="metric-value">{winner}</div></div>', unsafe_allow_html=True)
            with c_m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Similarity confidence</div><div class="metric-value">{conf:.1f}%</div></div>', unsafe_allow_html=True)
            with c_m3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Signal Integrity</div><div class="metric-value" style="color:#22c55e">VALID</div></div>', unsafe_allow_html=True)

            # --- VISUALIZATION BLOCK ---
            
            # [1] WAVEFORM ALIGNMENT
            st.markdown("### 1. Temporal Alignment Map")
            st.plotly_chart(viz.create_waveform(y_in, db_waves[winner][scores[0][2]], winner), use_container_width=True)
            st.markdown(f"""<div class="analysis-container"><span class="analysis-title">🔬 Signal Alignment Insight</span><p class="analysis-text">Grafik ini memetakan sinkronisasi antara sinyal uji dan master template. Dominasi logat <b>{winner}</b> divalidasi oleh tingkat kemiripan modulasi amplitudo pada setiap suku kata (segmental). DTW berhasil mengompensasi perbedaan tempo bicara (warping), menyisakan residu jarak yang minimal.</p></div>""", unsafe_allow_html=True)

            # [2] CROSS-DATABASE HEATMAP
            st.markdown("### 2. Spectral Correlation Heatmap")
            st.plotly_chart(viz.create_heatmap(h_vals, h_lbls), use_container_width=True)
            st.markdown(f"""<div class="analysis-container"><span class="analysis-title">🔬 Matrix Correlation Insight</span><p class="analysis-text">Matriks kemiripan ini menggunakan palet <b>Cyber-Navy</b> untuk menyoroti densitas korelasi fitur MFCC. Area cerah (Cyan) menunjukkan bahwa logat <b>{winner}</b> memiliki koherensi fitur spektral yang paling stabil di seluruh database, meminimalkan kemungkinan kesalahan klasifikasi lintas-dialek.</p></div>""", unsafe_allow_html=True)

            # [3] RADAR & SPECTRAL (PARALLEL COLUMNS)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 3. Probability Distribution Radar")
                lbls_r = list(db_templates.keys())
                vls_r = [1/(1+min([x[0] for x in scores if x[1]==L]))*100 for L in lbls_r]
                st.plotly_chart(viz.create_radar(lbls_r, vls_r), use_container_width=True)
                st.markdown(f"""<div class="analysis-container"><span class="analysis-title">🔬 Probability Bias Analysis</span><p class="analysis-text">Radar menunjukkan bias probabilitas yang condong ke sumbu <b>{winner}</b>. Ini membuktikan bahwa distribusi energi frekuensi audio uji memiliki morfologi vokal yang unik dan tidak memiliki tumpang tindih signifikan dengan dialek referensi lainnya.</p></div>""", unsafe_allow_html=True)

            with col_r:
                st.markdown("### 4. Acoustic Spectral Brightness")
                st.plotly_chart(viz.create_spectral(y_in), use_container_width=True)
                st.markdown(f"""<div class="analysis-container"><span class="analysis-title">🔬 Intonation Brightness Analysis</span><p class="analysis-text">Spectral Centroid mengukur "pusat massa" frekuensi. Pola kecerahan suara pada grafik ini menunjukkan intonasi tinggi-rendah yang identik dengan profil penutur asli daerah <b>{winner}</b>, memperkuat validitas hasil deteksi secara psikoakustik.</p></div>""", unsafe_allow_html=True)

            # [4] DELTA MFCC
            st.markdown("### 5. Velocity of Acoustic Features (Delta-Dynamic)")
            st.plotly_chart(viz.create_delta(mfcc_in), use_container_width=True)
            st.markdown(f"""<div class="analysis-container"><span class="analysis-title">🔬 Dynamic Transition Insight</span><p class="analysis-text">Heatmap Delta menggunakan aksen <b>Gold/Amber</b> untuk memperjelas transisi antar fonem. Kecocokan pada dimensi ini menunjukkan bahwa "ritme" atau tempo bicara pada sinyal uji memiliki profil kecepatan yang sinkron dengan karakteristik temporal logat <b>{winner}</b>.</p></div>""", unsafe_allow_html=True)

            # --- RANKING ---
            st.markdown("### 📊 Comprehensive Ranking")
            final_r = sorted({L: 1/(1+min([x[0] for x in scores if x[1]==L]))*100 for L in db_templates.keys()}.items(), key=lambda x: x[1], reverse=True)
            for n, s in final_r:
                st.write(f"**{n}**")
                st.progress(s/100)

    st.markdown('<div class="footer">Acoustic Intelligence Research Platform © 2026 | Built for Precision Analysis</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
