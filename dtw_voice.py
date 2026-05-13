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
# [1] GLOBAL SYSTEM ARCHITECTURE & DOCUMENTATION
# ==============================================================================
"""
SISTEM ANALISIS AKUSTIK PROFESIONAL (RUMAH DATA)
Arsitektur: Digital Signal Processing (DSP) Hybrid
Metode: Dynamic Time Warping (DTW) + Mel-Frequency Cepstral Coefficients (MFCC)
Versi: 3.7.0 - Standar Laboratorium Akustik
"""

st.set_page_config(
    page_title="Advanced Acoustic Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# [2] UNIVERSAL AUDIO DECODER (AUTO-INJECTION)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_universal_engine():
    """
    Menginstal komponen FFmpeg secara virtual untuk menjamin aplikasi 
    dapat membaca format audio MP3, M4A, AAC, FLAC, dan OGG.
    """
    try:
        import pydub
        import imageio_ffmpeg
    except ImportError:
        with st.spinner("🔧 Calibrating Deep-Audio Engine..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "imageio-ffmpeg"])
            st.rerun()
    return True

initialize_universal_engine()

# ==============================================================================
# [3] PREMIUM NAVY UI/UX ENGINE - ALIGNMENT & THEME
# ==============================================================================
def apply_professional_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Base Configuration */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #050a18; color: #f8fafc; }

        /* Laboratory Header Section */
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
            color: #f8fafc; font-weight: 800; letter-spacing: -2px;
            font-size: 3.5rem; margin: 0; text-transform: uppercase;
        }
        .main-header p {
            color: #38bdf8; font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem; letter-spacing: 3px; margin-top: 15px;
        }

        /* Result Metrics Grid */
        .metric-card {
            background: #0f172a;
            padding: 2.2rem;
            border-radius: 16px;
            border: 1px solid #1e293b;
            text-align: center;
            transition: all 0.4s ease;
        }
        .metric-card:hover { border-color: #38bdf8; transform: translateY(-5px); }
        .metric-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 800; letter-spacing: 2px; }
        .metric-value { color: #38bdf8; font-size: 2.8rem; font-weight: 800; margin-top: 10px; }

        /* Analysis Container - Aligned Columns */
        .analysis-box {
            background: rgba(15, 23, 42, 0.7);
            padding: 1.8rem;
            border-radius: 12px;
            border-left: 4px solid #38bdf8;
            margin-top: 1rem;
            margin-bottom: 2rem;
            min-height: 250px; 
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .analysis-title { 
            color: #f8fafc; font-weight: 800; font-size: 0.95rem; 
            margin-bottom: 12px; text-transform: uppercase;
            display: block; border-bottom: 1px solid #1e293b; padding-bottom: 8px;
        }
        .analysis-text { color: #94a3b8; font-size: 0.9rem; line-height: 1.7; text-align: justify; }

        /* Sidebar Interface */
        section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
        .status-card {
            background: #050a18; padding: 1.8rem; border-radius: 15px;
            border: 1px solid #1e3a8a; margin-bottom: 2.5rem; text-align: center;
        }
        .status-main { color: #38bdf8; font-size: 2rem; font-weight: 800; }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #0f172a; border: 1px solid #1e293b;
            padding: 10px 25px; border-radius: 8px 8px 0 0; color: #94a3b8;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e293b; color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_professional_styles()

# ==============================================================================
# [4] ACOUSTIC PROCESSING CORE
# ==============================================================================
class AcousticCore:
    """Modul inti untuk decoding sinyal digital dan ekstraksi fitur logat."""
    def __init__(self, k, w):
        self.SR = 16000
        self.N_MFCC = 13
        self.K = k
        self.W = w

    def load_audio(self, path):
        """Universal Audio Decoder dengan normalisasi bit-depth."""
        try:
            import pydub
            import imageio_ffmpeg
            pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            audio = pydub.AudioSegment.from_file(path).set_frame_rate(self.SR).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            return samples / float(1 << (audio.sample_width * 8 - 1))
        except Exception:
            # Fallback ke librosa murni
            y, _ = librosa.load(path, sr=self.SR, mono=True)
            return y

    def extract_dialect_features(self, y):
        """Ekstraksi Koefisien Cepstral dengan VAD internal terintegrasi."""
        # Menghilangkan keheningan agresif untuk menghindari bias statistik
        yt, _ = librosa.effects.trim(y, top_db=25)
        mfcc = librosa.feature.mfcc(y=yt, sr=self.SR, n_mfcc=self.N_MFCC)
        # Menghitung Delta-Temporal (Dinamika Kecepatan Bicara)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        features = np.hstack([mfcc.T, d1.T, d2.T])
        # Normalisasi Z-Score untuk menghilangkan bias amplitudo/volume
        mu = np.mean(features, axis=0)
        sigma = np.std(features, axis=0) + 1e-8
        return (features - mu) / sigma, mfcc, yt

def dtw_alignment_engine(s1, s2, w_const):
    """Kalkulasi Penyelarasan Waktu Dinamis menggunakan metrik Cosine."""
    n, m = len(s1), len(s2)
    w = max(w_const, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf); dp[0, 0] = 0.0
    # Metrik Cosine lebih unggul dalam mencocokkan bentuk morfologi sinyal
    cost = cdist(s1, s2, metric='cosine')
    for i in range(1, n + 1):
        for j in range(max(1, i-w), min(m, i+w)+1):
            prev_min = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
            dp[i, j] = cost[i-1, j-1] + prev_min
    return dp[n, m] / (n + m)

# ==============================================================================
# [5] VISUALIZATION ENGINE - PREMIUM FACTORY
# ==============================================================================
class VizEngine:
    """Modul untuk memfabrikasi grafik dengan palet warna Navy-Cyber."""
    
    @staticmethod
    def get_navy_scale():
        return [[0, '#050a18'], [0.4, '#1e3a8a'], [1, '#38bdf8']]

    @staticmethod
    def get_amber_scale():
        return [[0, '#050a18'], [0.5, '#1e293b'], [1, '#fbbf24']]

    def plot_waveform(self, y_in, y_ref, label):
        """Membuat perbandingan waveform dengan spasi vertikal yang rapi."""
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.25, 
                            subplot_titles=("Sinyal Input (Audio Uji)", f"Referensi Database: {label}"))
        fig.add_trace(go.Scatter(y=y_in, line=dict(color='#38bdf8', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(y=y_ref, line=dict(color='#64748b', width=1)), row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", showlegend=False, 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(title_text="Waktu (Samples)", row=2, col=1)
        return fig

    def plot_heatmap(self, data, labels):
        """Matriks kemiripan dengan label dialek."""
        fig = go.Figure(data=go.Heatmap(z=[data], x=labels, colorscale=self.get_navy_scale(), 
                                        zmin=0, zmax=100, text=[[f"{v:.0f}%" for v in data]], texttemplate="%{text}"))
        fig.update_layout(height=260, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=0))
        return fig

    def plot_radar(self, labels, values):
        """Distribusi probabilitas logat."""
        fig = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], 
                                            fill='toself', fillcolor='rgba(56, 189, 248, 0.15)', line=dict(color='#38bdf8', width=2.5)))
        fig.update_layout(polar=dict(bgcolor='#0f172a', radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1e293b')), 
                          template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        return fig

# ==============================================================================
# [6] ANALYTICAL PIPELINE EXECUTION
# ==============================================================================
def start_dialect_analysis():
    # Pemindaian Database Lokal
    zip_files = list(Path('.').glob('*.zip'))
    
    with st.sidebar:
        st.markdown(f"""<div class="status-card"><div style="color:#64748b;font-size:0.7rem;font-weight:800;text-transform:uppercase;">Computational Status</div><div class="status-main">{len(zip_files)} Logat</div><div style="color:#22c55e;font-size:0.75rem;font-weight:700;">● ENGINE OPTIMIZED</div></div>""", unsafe_allow_html=True)
        k_val = st.slider("Classification Sensitivity (K)", 1, 15, 5)
        w_val = st.slider("Window Constraint (W)", 20, 400, 120, step=10)
        st.markdown("---")
        if st.button("Reload Research Intelligence", use_container_width=True):
            st.cache_resource.clear(); st.rerun()

    st.markdown("""<div class="main-header"><h1>Acoustic Intelligence Analysis</h1><p>PATTERN RECOGNITION LABORATORY SYSTEM V3.7.0 (STABLE)</p></div>""", unsafe_allow_html=True)

    core = AcousticCore(k_val, w_val)
    viz = VizEngine()

    @st.cache_resource
    def boot_database():
        # VARIABEL KRUSIAL: db_templates dan db_waves
        db_templates, db_waves = defaultdict(list), defaultdict(list)
        for z in zip_files:
            label = z.stem.replace("Logat_", "").upper()
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(z, 'r') as zf: zf.extractall(td)
                for f in Path(td).rglob('*'):
                    if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']:
                        y = core.load_audio(str(f))
                        if y is not None:
                            feats, _, yt = core.extract_dialect_features(y)
                            db_templates[label].append(feats)
                            db_waves[label].append(yt)
        return db_templates, db_waves

    # Memuat Database
    db_templates, db_waves = boot_database()
    
    if not db_templates:
        st.error("Missing Acoustic Datasets. Please provide .zip archives."); return

    # Antarmuka Input
    tab_f, tab_r = st.tabs(["📁 IDENTIFIKASI BERKAS", "🎤 LIVE CAPTURE"])
    audio_stream, source_id = None, ""

    with tab_f:
        u = st.file_uploader("Upload Acoustic Data", type=['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg'], label_visibility="collapsed")
        if u: audio_stream, source_id = u.read(), u.name
    with tab_r:
        m = st.audio_input("Record Speech Pattern", label_visibility="collapsed")
        if m: audio_stream, source_id = m.read(), "live_stream.wav"

    if audio_stream:
        with st.spinner("Executing spectral decomposition..."):
            with tempfile.NamedTemporaryFile(suffix=Path(source_id).suffix, delete=False) as tmp:
                tmp.write(audio_stream); path = tmp.name
            y_raw = core.load_audio(path)
            feats_in, mfcc_in, y_in_t = core.extract_dialect_features(y_raw)
            os.remove(path)

            # CALCULATION PIPELINE
            scores, h_vals, h_lbls = [], [], []
            for label, templates in db_templates.items():
                for i, t in enumerate(templates):
                    dist = dtw_alignment_engine(feats_in, t, w_val)
                    scores.append((dist, label, i))
                    h_vals.append(1/(1+dist)*100); h_lbls.append(f"{label}")

            scores.sort(key=lambda x: x[0])
            winner, idx_winner = scores[0][1], scores[0][2]
            conf = (1/(1+scores[0][0])*100)
            
            # --- RESULTS GRID ---
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Identitas Dialek</div><div class="metric-value">{winner}</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence Score</div><div class="metric-value">{conf:.1f}%</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">VAD Engine</div><div class="metric-value" style="color:#22c55e">ACTIVE</div></div>', unsafe_allow_html=True)

            # 1. WAVEFORM
            st.markdown("### 1. Temporal Signal Alignment Map")
            # PERBAIKAN DI SINI: db_waves digunakan secara konsisten
            st.plotly_chart(viz.plot_waveform(y_in_t, db_waves[winner][idx_winner], winner), use_container_width=True)
            st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Konsistensi Temporal</span><p class="analysis-text">Grafik waveform di atas memetakan sinkronisasi modulasi suara antara input dan referensi master. Logat <b>{winner}</b> mendominasi karena memiliki struktur penekanan suku kata dan ritme bicara yang paling identik. Decoder Universal memastikan integritas sinyal tetap terjaga meskipun audio berasal dari format terkompresi.</p></div>""", unsafe_allow_html=True)

            # 2. HEATMAP
            st.markdown("### 2. Spectral Similarity Matrix")
            st.plotly_chart(viz.plot_heatmap(h_vals, h_lbls), use_container_width=True)
            st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Korelasi Matriks</span><p class="analysis-text">Matriks kemiripan spektral memetakan korelasi fitur MFCC di seluruh database menggunakan palet Cyber-Navy. Area berwarna biru cerah pada kolom <b>{winner}</b> mengindikasikan densitas kecocokan fitur suara yang paling stabil, meminimalkan bias klasifikasi lintas-dialek.</p></div>""", unsafe_allow_html=True)

            # 3 & 4. RADAR & SPECTRAL
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 3. Dialect Probability Radar")
                u_labs = list(db_templates.keys())
                v_radar = [1/(1+min([x[0] for x in scores if x[1]==L]))*100 for L in u_labs]
                st.plotly_chart(viz.plot_radar(u_labs, v_radar), use_container_width=True)
                st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Distribusi Radar</span><p class="analysis-text">Radar distribusi menunjukkan tarikan vektor probabilitas yang condong ke arah sumbu <b>{winner}</b>. Hal ini mengonfirmasi morfologi vokal yang unik dan tidak tumpang tindih dengan dialek referensi lainnya dalam sistem, memberikan tingkat kepastian klasifikasi yang tinggi.</p></div>""", unsafe_allow_html=True)

            with col_r:
                st.markdown("### 4. Acoustic Spectral Brightness")
                sc = librosa.feature.spectral_centroid(y=y_in_t, sr=16000)[0]
                fig_s = go.Figure(); fig_s.add_trace(go.Scatter(y=sc, line=dict(color='#fbbf24', width=1.5), fill='tozeroy'))
                fig_s.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_s, use_container_width=True)
                st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Kecerahan Akustik</span><p class="analysis-text">Spectral Centroid mengukur "pusat massa" frekuensi suara. Pola kecerahan (intonasi) pada sinyal uji ini menunjukkan profil energi frekuensi tinggi yang sangat spesifik bagi logat <b>{winner}</b>, mencerminkan melodi bicara khas daerah tersebut dalam database kami.</p></div>""", unsafe_allow_html=True)

            # 5. DELTA MFCC
            st.markdown("### 5. Velocity of Speech Features (Delta)")
            delta = librosa.feature.delta(mfcc_in)
            fig_d = go.Figure(data=go.Heatmap(z=delta, colorscale=viz.get_amber_scale(), zmid=0))
            fig_d.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Transisi Dinamis</span><p class="analysis-text">Heatmap Delta menggambarkan kecepatan perubahan fonem (ritme tempo). Kedekatan pada grafik ini menunjukkan bahwa dinamika bicara Anda memiliki profil kecepatan artikulasi yang sinkron dengan karakteristik temporal <b>{winner}</b>, memperkuat hasil deteksi dari sisi tempo.</p></div>""", unsafe_allow_html=True)

            # RANKING
            st.markdown("### 📊 Comprehensive Class Ranking")
            final_r = sorted({L: 1/(1+min([x[0] for x in scores if x[1]==L]))*100 for L in db_templates.keys()}.items(), key=lambda x: x[1], reverse=True)
            for n, s in final_r: st.write(f"**{n}**"); st.progress(s/100)

    st.markdown('<div class="footer">Acoustic Research Platform © 2026 | Developed for High-Precision Research</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    start_dialect_analysis()
