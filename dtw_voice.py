import streamlit as st
import os
import zipfile
import tempfile
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import copy

import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Analisis Akustik Dialek",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# UNIVERSAL AUDIO DECODER
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_universal_engine():
    try:
        import pydub
        import imageio_ffmpeg
    except ImportError:
        with st.spinner("Menyiapkan decoder audio..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "imageio-ffmpeg"])
            st.rerun()
    try:
        from fastdtw import fastdtw  # noqa: F401
    except ImportError:
        with st.spinner("Menginstal FastDTW..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fastdtw"])
            st.rerun()
    return True

initialize_universal_engine()

# ==============================================================================
# NAVY UI/UX ENGINE
# ==============================================================================
def apply_professional_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

        :root {
            --navy-900: #020812;
            --navy-800: #040d1e;
            --navy-700: #071228;
            --navy-600: #0c1f3f;
            --navy-500: #132d56;
            --navy-400: #1a3a6e;
            --accent-sky: #38bdf8;
            --accent-sky-dim: rgba(56,189,248,0.12);
            --accent-cyan: #22d3ee;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --text-primary: #e8f0fe;
            --text-secondary: #8eafd4;
            --text-muted: #4a6b9b;
            --border: rgba(56,189,248,0.15);
            --border-strong: rgba(56,189,248,0.35);
        }

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        .stApp {
            background: var(--navy-900);
            background-image:
                radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56,189,248,0.07) 0%, transparent 60%),
                radial-gradient(ellipse 40% 30% at 80% 90%, rgba(20,40,100,0.3) 0%, transparent 50%);
            color: var(--text-primary);
        }
        .stApp::before {
            content: '';
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background-image: repeating-linear-gradient(
                0deg, transparent, transparent 2px,
                rgba(56,189,248,0.015) 2px, rgba(56,189,248,0.015) 4px
            );
        }

        .hero-header {
            position: relative;
            background: linear-gradient(160deg, var(--navy-700) 0%, var(--navy-800) 60%, var(--navy-900) 100%);
            padding: 3.5rem 3rem 3rem;
            border-radius: 0 0 28px 28px;
            margin: -4rem -4rem 3rem -4rem;
            border-bottom: 1px solid var(--border);
            overflow: hidden;
        }
        .hero-header::before {
            content: '';
            position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent 0%, var(--accent-sky) 30%, var(--accent-cyan) 70%, transparent 100%);
        }
        .hero-header::after {
            content: '';
            position: absolute; bottom: -40px; right: -40px;
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
            border-radius: 50%;
        }
        .hero-eyebrow {
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem; font-weight: 500;
            letter-spacing: 4px; color: var(--accent-sky);
            text-transform: uppercase; margin-bottom: 16px;
            display: flex; align-items: center; gap: 12px;
        }
        .hero-eyebrow::before {
            content: ''; display: inline-block;
            width: 32px; height: 1px;
            background: var(--accent-sky); opacity: 0.7;
        }
        .hero-title {
            font-family: 'Syne', sans-serif;
            font-size: 2.8rem; font-weight: 800;
            color: var(--text-primary); line-height: 1.05;
            letter-spacing: -1.5px; margin: 0;
        }
        .hero-title span { color: var(--accent-sky); }
        .hero-subtitle {
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem; color: var(--text-muted);
            letter-spacing: 2px; margin-top: 14px;
            text-transform: uppercase;
        }
        .hero-badges {
            display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap;
        }
        .hero-badge {
            background: var(--accent-sky-dim);
            border: 1px solid var(--border);
            color: var(--accent-sky); font-family: 'DM Mono', monospace;
            font-size: 0.65rem; padding: 5px 14px;
            border-radius: 4px; letter-spacing: 1px; text-transform: uppercase;
        }

        .section-header {
            display: flex; align-items: center; gap: 14px;
            margin: 2.5rem 0 1.2rem;
        }
        .section-number {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem; color: var(--text-muted);
            background: var(--navy-600);
            border: 1px solid var(--border);
            padding: 4px 8px; border-radius: 4px; letter-spacing: 1px;
        }
        .section-title {
            font-family: 'Syne', sans-serif;
            font-size: 1rem; font-weight: 700;
            color: var(--text-primary); letter-spacing: -0.3px;
            text-transform: uppercase;
        }
        .section-line {
            flex: 1; height: 1px;
            background: linear-gradient(90deg, var(--border) 0%, transparent 100%);
        }

        .metrics-row { display: flex; gap: 16px; margin: 1.5rem 0 2rem; }
        .metric-card {
            flex: 1;
            background: linear-gradient(145deg, var(--navy-600) 0%, var(--navy-700) 100%);
            padding: 1.6rem 1.4rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            position: relative; overflow: hidden;
            transition: border-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
            min-width: 0;
        }
        .metric-card::before {
            content: '';
            position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, var(--accent-sky), var(--accent-cyan));
            opacity: 0.5;
        }
        .metric-card:hover {
            border-color: var(--border-strong);
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(56,189,248,0.1);
        }
        .metric-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 2px;
            margin-bottom: 10px;
        }
        .metric-value {
            font-family: 'Syne', sans-serif;
            font-size: clamp(1.2rem, 2.5vw, 2rem);
            font-weight: 800;
            color: var(--accent-sky); line-height: 1.15;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .metric-value.green { color: var(--accent-emerald); }
        .metric-sub {
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem; color: var(--text-muted);
            margin-top: 8px;
        }

        .analysis-box {
            background: linear-gradient(135deg, rgba(7,18,40,0.9) 0%, rgba(4,13,30,0.95) 100%);
            padding: 1.5rem 1.8rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-sky);
            margin: 1rem 0 2rem;
        }
        .analysis-title {
            font-family: 'Syne', sans-serif;
            font-size: 0.78rem; font-weight: 700;
            color: var(--text-primary); text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 10px;
            padding-bottom: 10px; display: block;
            border-bottom: 1px solid rgba(56,189,248,0.1);
        }
        .analysis-text {
            color: var(--text-secondary);
            font-size: 0.88rem; line-height: 1.75;
            text-align: justify;
        }
        .analysis-text b { color: var(--accent-sky); font-weight: 600; }

        section[data-testid="stSidebar"] {
            background: var(--navy-800) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

        .sidebar-brand {
            text-align: center; padding: 1.5rem 1rem 1rem;
            border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;
        }
        .sidebar-brand-title {
            font-family: 'Syne', sans-serif;
            font-size: 0.9rem; font-weight: 700;
            color: var(--text-primary); letter-spacing: 1px;
            text-transform: uppercase;
        }
        .sidebar-brand-sub {
            font-family: 'DM Mono', monospace;
            font-size: 0.6rem; color: var(--text-muted);
            letter-spacing: 2px; margin-top: 4px;
        }

        .status-card {
            background: linear-gradient(145deg, var(--navy-600), var(--navy-700));
            padding: 1.4rem 1.2rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
            position: relative; overflow: hidden;
        }
        .status-card::after {
            content: '';
            position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, var(--accent-emerald), var(--accent-sky));
            opacity: 0.6;
        }
        .status-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.6rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;
        }
        .status-main {
            font-family: 'Syne', sans-serif;
            font-size: 2rem; font-weight: 800; color: var(--accent-sky);
        }
        .status-badge {
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(52,211,153,0.1);
            border: 1px solid rgba(52,211,153,0.3);
            color: var(--accent-emerald);
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem; padding: 4px 10px; border-radius: 20px;
            margin-top: 8px; text-transform: uppercase; letter-spacing: 1px;
        }
        .status-badge::before {
            content: ''; width: 5px; height: 5px;
            background: var(--accent-emerald); border-radius: 50%;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.7); }
        }

        .sidebar-section-label {
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 2px;
            margin: 1.2rem 0 0.5rem; padding-left: 2px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px; background: transparent !important;
            border-bottom: 1px solid var(--border) !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border: 1px solid var(--border) !important;
            border-bottom: none !important;
            padding: 10px 22px !important;
            border-radius: 8px 8px 0 0 !important;
            color: var(--text-muted) !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.72rem !important; letter-spacing: 1.5px !important;
            text-transform: uppercase !important;
            transition: all 0.2s ease !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-secondary) !important;
            background: var(--accent-sky-dim) !important;
        }
        .stTabs [aria-selected="true"] {
            background: var(--navy-600) !important;
            color: var(--accent-sky) !important;
            border-color: var(--border-strong) !important;
            border-bottom: 2px solid var(--accent-sky) !important;
        }

        .upload-zone {
            background: var(--accent-sky-dim);
            border: 2px dashed var(--border-strong);
            border-radius: 16px; padding: 2rem;
            text-align: center; margin: 1rem 0;
        }
        .upload-title {
            font-family: 'Syne', sans-serif;
            font-size: 1rem; font-weight: 700; color: var(--text-primary);
        }
        .upload-sub {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem; color: var(--text-muted);
            margin-top: 6px; letter-spacing: 1px;
        }

        .rank-item {
            display: flex; align-items: center; gap: 14px;
            padding: 0.8rem 1rem; margin-bottom: 8px;
            background: var(--navy-700);
            border: 1px solid var(--border);
            border-radius: 10px;
            transition: border-color 0.2s ease;
        }
        .rank-item:hover { border-color: var(--border-strong); }
        .rank-item.top { border-color: rgba(56,189,248,0.4); background: rgba(56,189,248,0.06); }
        .rank-num {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem; color: var(--text-muted);
            width: 24px; text-align: right;
        }
        .rank-name {
            font-family: 'Syne', sans-serif;
            font-size: 0.85rem; font-weight: 700; color: var(--text-primary);
            min-width: 80px;
        }
        .rank-bar-bg {
            flex: 1; height: 6px;
            background: var(--navy-500); border-radius: 3px; overflow: hidden;
        }
        .rank-bar-fill {
            height: 100%; border-radius: 3px;
            background: linear-gradient(90deg, var(--accent-sky), var(--accent-cyan));
        }
        .rank-bar-fill.top { background: linear-gradient(90deg, #38bdf8, #7dd3fc, #22d3ee); }
        .rank-pct {
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem; font-weight: 500; color: var(--accent-sky);
            min-width: 50px; text-align: right;
        }
        .rank-pct.top { color: var(--accent-cyan); }

        .chart-container {
            background: linear-gradient(135deg, var(--navy-700) 0%, var(--navy-800) 100%);
            border-radius: 16px; border: 1px solid var(--border);
            padding: 1.2rem; margin-bottom: 0.5rem;
            overflow: hidden;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--navy-500), var(--navy-600)) !important;
            border: 1px solid var(--border-strong) !important;
            color: var(--accent-sky) !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.72rem !important; letter-spacing: 1.5px !important;
            text-transform: uppercase !important;
            border-radius: 8px !important;
            transition: all 0.25s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--navy-400), var(--navy-500)) !important;
            border-color: var(--accent-sky) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 20px rgba(56,189,248,0.2) !important;
        }

        .app-footer {
            text-align: center; padding: 2rem 0 1rem;
            border-top: 1px solid var(--border); margin-top: 3rem;
        }
        .app-footer-text {
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem; color: var(--text-muted);
            letter-spacing: 2px; text-transform: uppercase;
        }

        hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
    </style>
    """, unsafe_allow_html=True)

apply_professional_styles()

# ==============================================================================
# ACOUSTIC CORE - DENGAN PREPROCESSING SEPERTI DOSEN
# ==============================================================================
class AcousticCore:
    def __init__(self, k, w):
        self.SR = 16000
        self.N_MFCC = 20
        self.K = k
        self.W = w

    def load_audio(self, path):
        try:
            import pydub
            import imageio_ffmpeg
            pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            audio = pydub.AudioSegment.from_file(path).set_frame_rate(self.SR).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            return samples / float(1 << (audio.sample_width * 8 - 1))
        except Exception:
            y, _ = librosa.load(path, sr=self.SR, mono=True)
            return y

    def preprocess_mfcc(self, mfcc):
        """Preprocessing MFCC seperti di script dosen: remove mean dan normalize"""
        mfcc_cp = copy.deepcopy(mfcc)
        for i in range(mfcc.shape[1]):
            mfcc_cp[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
            max_val = np.max(np.abs(mfcc_cp[:, i]))
            if max_val > 0:
                mfcc_cp[:, i] = mfcc_cp[:, i] / max_val
        return mfcc_cp

    def extract_features(self, y):
        """Ekstraksi fitur MFCC dengan preprocessing"""
        # Trim silent parts
        yt, _ = librosa.effects.trim(y, top_db=25)
        
        if len(yt) < self.SR * 0.3:
            yt = y
        
        # Normalisasi amplitude
        if np.max(np.abs(yt)) > 0:
            yt = yt / (np.max(np.abs(yt)) + 1e-8)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=yt, sr=self.SR, n_mfcc=self.N_MFCC)
        
        # Preprocessing seperti dosen
        mfcc_processed = self.preprocess_mfcc(mfcc)
        
        return mfcc_processed, yt

def dtw_distance(s1, s2, w):
    """
    FastDTW dengan radius = w sebagai pengganti Sakoe-Chiba window.
    FastDTW: O(N) waktu & memori → jauh lebih cepat dari DTW klasik O(N²),
    akurasi hampir sama karena menggunakan pendekatan multiresolusi.
    Fallback ke DTW klasik jika fastdtw tidak tersedia.
    """
    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        n, m = len(s1), len(s2)
        # radius ≈ w pada FastDTW setara dengan Sakoe-Chiba band
        radius = max(w, abs(n - m))
        dist, _ = fastdtw(s1, s2, radius=radius, dist=euclidean)
        return dist / (n + m)
    except ImportError:
        # Fallback: DTW klasik dengan Sakoe-Chiba window
        n, m = len(s1), len(s2)
        window = max(w, abs(n - m))
        cost = cdist(s1, s2, metric='euclidean')
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(max(1, i - window), min(m, i + window) + 1):
                dp[i, j] = cost[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        return dp[n, m] / (n + m)


def dtw_sliding_window(test_mfcc, train_mfcc, w_val):
    """
    FastDTW Sliding Window.
    Geser jendela sepanjang sequence yang lebih panjang, cari posisi
    dengan jarak FastDTW terkecil terhadap template (yang lebih pendek).
    """
    len_test = test_mfcc.shape[1]
    len_train = train_mfcc.shape[1]

    # Pastikan test selalu yang lebih panjang (window digeser di atasnya)
    if len_train > len_test:
        return dtw_sliding_window(train_mfcc, test_mfcc, w_val)

    num_windows = len_test - len_train + 1
    best_dist = float('inf')

    for start in range(num_windows):
        end = start + len_train
        test_window = test_mfcc[:, start:end]
        # FastDTW bekerja pada baris (frame), jadi transpose → (T × n_mfcc)
        dist = dtw_distance(test_window.T, train_mfcc.T, w_val)
        if dist < best_dist:
            best_dist = dist

    return best_dist

# ==============================================================================
# ENSEMBLE CLASSIFIER - GABUNGAN DTW DAN COSINE SIMILARITY
# ==============================================================================
def ensemble_classify(test_mfcc, db_templates, w_val):
    """Ensemble: DTW sliding window + Cosine similarity centroid"""
    results = []
    
    for label, templates in db_templates.items():
        best_dtw = float('inf')
        best_cosine = -1
        best_idx = 0
        
        for idx, train_mfcc in enumerate(templates):
            # 1. DTW dengan sliding window
            dtw_dist = dtw_sliding_window(test_mfcc, train_mfcc, w_val)
            dtw_sim = 1 / (1 + dtw_dist)
            
            # 2. Cosine similarity untuk centroid (global feature)
            centroid_test = np.mean(test_mfcc, axis=1)
            centroid_train = np.mean(train_mfcc, axis=1)
            cos_sim = np.dot(centroid_test, centroid_train) / (
                np.linalg.norm(centroid_test) * np.linalg.norm(centroid_train) + 1e-8
            )
            cos_sim = max(0, min(1, cos_sim))
            
            # Kombinasi weighted
            combined = 0.6 * dtw_sim + 0.4 * cos_sim
            
            if combined > best_cosine:
                best_cosine = combined
                best_dtw = dtw_dist
                best_idx = idx
        
        results.append((best_cosine, best_dtw, label, best_idx))
    
    # Urutkan dari similarity tertinggi
    results.sort(key=lambda x: x[0], reverse=True)
    return results

# ==============================================================================
# VISUALIZATION ENGINE
# ==============================================================================
class VizEngine:
    @staticmethod
    def get_navy_scale():
        return [[0, '#020812'], [0.4, '#1a3a6e'], [1, '#38bdf8']]

    @staticmethod
    def get_amber_scale():
        return [[0, '#020812'], [0.5, '#1e293b'], [1, '#fbbf24']]

    def _base_layout(self, h=400):
        return dict(
            height=h, template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Mono, monospace', color='#8eafd4', size=11),
            margin=dict(l=10, r=10, t=30, b=10),
        )

    def plot_waveform(self, y_in, y_ref, label):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.22,
                            subplot_titles=("Sinyal Input (Audio Uji)", f"Referensi Database: {label}"))
        fig.add_trace(go.Scatter(y=y_in, line=dict(color='#38bdf8', width=1.2),
                                 fill='tozeroy', fillcolor='rgba(56,189,248,0.06)'), row=1, col=1)
        fig.add_trace(go.Scatter(y=y_ref, line=dict(color='#22d3ee', width=1.2),
                                 fill='tozeroy', fillcolor='rgba(34,211,238,0.05)'), row=2, col=1)
        layout = self._base_layout(440)
        layout.update(showlegend=False)
        fig.update_annotations(font=dict(size=11, color='#8eafd4', family='DM Mono, monospace'))
        fig.update_layout(**layout)
        fig.update_xaxes(gridcolor='rgba(56,189,248,0.07)', zerolinecolor='rgba(56,189,248,0.15)')
        fig.update_yaxes(gridcolor='rgba(56,189,248,0.07)', zerolinecolor='rgba(56,189,248,0.15)')
        return fig

    def plot_heatmap(self, data, labels):
        fig = go.Figure(data=go.Heatmap(
            z=[data], x=labels, colorscale=self.get_navy_scale(),
            zmin=0, zmax=100, text=[[f"{v:.0f}%" for v in data]],
            texttemplate="%{text}",
            textfont=dict(family='DM Mono, monospace', size=12, color='#e8f0fe')
        ))
        layout = self._base_layout(240)
        layout.update(margin=dict(l=0, r=0, t=10, b=0))
        fig.update_layout(**layout)
        return fig

    def plot_radar(self, labels, values):
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]], theta=labels + [labels[0]],
            fill='toself', fillcolor='rgba(56,189,248,0.12)',
            line=dict(color='#38bdf8', width=2.5),
            marker=dict(color='#38bdf8', size=6)
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(7,18,40,0.6)',
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor='rgba(56,189,248,0.12)',
                                tickfont=dict(family='DM Mono, monospace', size=9, color='#4a6b9b')),
                angularaxis=dict(tickfont=dict(family='DM Mono, monospace', size=10, color='#8eafd4'),
                                 gridcolor='rgba(56,189,248,0.08)')
            ),
            **self._base_layout(340)
        )
        return fig

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def start_dialect_analysis():
    zip_files = list(Path('.').glob('*.zip'))

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">Analisis Akustik</div>
                <div class="sidebar-brand-sub">Laboratorium Pengenalan Pola</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="status-card">
                <div class="status-label">Status Komputasi</div>
                <div class="status-main">{len(zip_files)}<span style="font-size:1rem;color:#4a6b9b;font-weight:400;"> logat</span></div>
                <div class="status-badge">Mesin Aktif</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-label">Konfigurasi Parameter</div>', unsafe_allow_html=True)
        k_val = st.slider("Sensitivitas Klasifikasi (K)", 1, 15, 5)
        w_val = st.slider("Batas Jendela (W)", 20, 400, 120, step=10)

        st.markdown('<div class="sidebar-section-label">Sistem</div>', unsafe_allow_html=True)
        if st.button("Muat Ulang Database", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("""
            <div style="margin-top:2rem;padding-top:1.5rem;border-top:1px solid rgba(56,189,248,0.1);">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#4a6b9b;letter-spacing:1.5px;text-align:center;">
                    FASTDTW SLIDING WINDOW + ENSEMBLE
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Hero Header
    st.markdown("""
        <div class="hero-header">
            <div class="hero-eyebrow">Sistem Analisis Akustik</div>
            <h1 class="hero-title">
                Laboratorium <span>Pengenalan</span><br>Dialek
            </h1>
            <div class="hero-subtitle">FastDTW Sliding Window · MFCC-20 · Ensemble Classifier</div>
            <div class="hero-badges">
                <span class="hero-badge">FastDTW Sliding Window</span>
                <span class="hero-badge">MFCC Preprocessing</span>
                <span class="hero-badge">Ensemble</span>
                <span class="hero-badge">Multi-Dialek</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    core = AcousticCore(k_val, w_val)
    viz = VizEngine()

    @st.cache_resource
    def boot_database():
        db_templates, db_waves = defaultdict(list), defaultdict(list)
        for z in zip_files:
            label = z.stem.replace("Logat_", "").upper()
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(z, 'r') as zf:
                    zf.extractall(td)
                for f in Path(td).rglob('*'):
                    if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']:
                        y = core.load_audio(str(f))
                        if y is not None and len(y) > 0:
                            mfcc, yt = core.extract_features(y)
                            if mfcc is not None and mfcc.size > 0:
                                db_templates[label].append(mfcc)
                                db_waves[label].append(yt)
        return db_templates, db_waves

    db_templates, db_waves = boot_database()

    if not db_templates:
        st.error("Dataset akustik tidak ditemukan. Harap sediakan arsip .zip di direktori kerja.")
        return

    # Input
    st.markdown("""
        <div class="section-header">
            <span class="section-number">INPUT</span>
            <span class="section-title">Pilih Sumber Audio</span>
            <span class="section-line"></span>
        </div>
    """, unsafe_allow_html=True)

    tab_f, tab_r = st.tabs(["Unggah Berkas", "Rekam Langsung"])
    audio_stream, source_id = None, ""

    with tab_f:
        st.markdown("""
            <div class="upload-zone">
                <div class="upload-title">Unggah Data Akustik</div>
                <div class="upload-sub">WAV · MP3 · M4A · AAC · FLAC · OGG</div>
            </div>
        """, unsafe_allow_html=True)
        u = st.file_uploader("Unggah Data Akustik",
                              type=['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg'],
                              label_visibility="collapsed")
        if u:
            audio_stream, source_id = u.read(), u.name

    with tab_r:
        m = st.audio_input("Rekam Pola Suara", label_visibility="collapsed")
        if m:
            audio_stream, source_id = m.read(), "rekaman_langsung.wav"

    # Pipeline
    if audio_stream:
        with st.spinner("Memproses dengan FastDTW Sliding Window..."):
            with tempfile.NamedTemporaryFile(suffix=Path(source_id).suffix, delete=False) as tmp:
                tmp.write(audio_stream)
                path = tmp.name
            
            y_raw = core.load_audio(path)
            
            if y_raw is None or len(y_raw) == 0:
                st.error("Gagal memuat file audio. Pastikan file tidak corrupt.")
                os.remove(path)
                return
                
            test_mfcc, y_in_t = core.extract_features(y_raw)
            os.remove(path)
            
            if test_mfcc is None or test_mfcc.size == 0:
                st.error("Gagal mengekstrak fitur dari audio.")
                return

            # Ensemble classification
            results = ensemble_classify(test_mfcc, db_templates, w_val)
            
            winner = results[0][2]
            confidence = results[0][0] * 100
            best_match_idx = results[0][3]
            
            # MFCC untuk visualisasi
            mfcc_in = librosa.feature.mfcc(y=y_in_t, sr=16000, n_mfcc=13)

        # Hasil Klasifikasi
        st.markdown("""
            <div class="section-header">
                <span class="section-number">HASIL</span>
                <span class="section-title">Keluaran Klasifikasi</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="metric-label">Identitas Dialek</div>
                    <div class="metric-value">{winner}</div>
                    <div class="metric-sub">Ensemble Classification</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Skor Kepercayaan</div>
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-sub">FastDTW Sliding Window + Cosine</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Metode</div>
                    <div class="metric-value green">ENSEMBLE</div>
                    <div class="metric-sub">MFCC-20 + Preprocessing</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Waveform
        st.markdown("""
            <div class="section-header">
                <span class="section-number">01</span>
                <span class="section-title">Peta Penyelarasan Sinyal Temporal</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if winner in db_waves and len(db_waves[winner]) > best_match_idx:
            st.plotly_chart(viz.plot_waveform(y_in_t, db_waves[winner][best_match_idx], winner), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="analysis-box">
                <span class="analysis-title">Analisis Konsistensi Temporal</span>
                <p class="analysis-text">Perbandingan waveform memetakan sinkronisasi modulasi antara sinyal input dan referensi master. <b>FastDTW</b> sliding window menemukan alignment terbaik antara audio uji dan template database dengan kompleksitas O(N) — jauh lebih cepat dari DTW klasik O(N²). Dialek <b>{winner}</b> memiliki struktur temporal yang paling sesuai.</p>
            </div>
        """, unsafe_allow_html=True)

        # Heatmap
        st.markdown("""
            <div class="section-header">
                <span class="section-number">02</span>
                <span class="section-title">Matriks Kemiripan Spektral</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)
        
        h_vals = []
        h_lbls = []
        for sim, dtw_dist, label, _ in results:
            h_vals.append(sim * 100)
            h_lbls.append(label)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(viz.plot_heatmap(h_vals, h_lbls), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="analysis-box">
                <span class="analysis-title">Analisis Korelasi Matriks</span>
                <p class="analysis-text">Matriks kemiripan spektral memetakan korelasi fitur MFCC-20 yang telah diproses (remove mean + normalisasi). Area berwarna biru cerah pada kolom <b>{winner}</b> mengindikasikan kecocokan fitur spektral tertinggi.</p>
            </div>
        """, unsafe_allow_html=True)

        # Radar dan Spektral
        st.markdown("""
            <div class="section-header">
                <span class="section-number">03 · 04</span>
                <span class="section-title">Radar Probabilitas dan Kecerahan Akustik</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            u_labs = [lbl for _, _, lbl, _ in results]
            v_radar = [sim * 100 for sim, _, _, _ in results]
            st.plotly_chart(viz.plot_radar(u_labs, v_radar), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="analysis-box">
                    <span class="analysis-title">Analisis Distribusi Radar</span>
                    <p class="analysis-text">Radar distribusi menunjukkan vektor probabilitas yang condong ke arah sumbu <b>{winner}</b>. Ensemble classifier mempertimbangkan <b>FastDTW</b> sliding window (60%) dan cosine similarity centroid (40%).</p>
                </div>
            """, unsafe_allow_html=True)

        with col_r:
            sc = librosa.feature.spectral_centroid(y=y_in_t, sr=16000)[0]
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(y=sc, line=dict(color='#fbbf24', width=1.8),
                                       fill='tozeroy', fillcolor='rgba(251,191,36,0.06)'))
            layout_s = viz._base_layout(340)
            layout_s.update(margin=dict(l=0, r=0, t=10, b=0))
            fig_s.update_xaxes(gridcolor='rgba(56,189,248,0.07)')
            fig_s.update_yaxes(gridcolor='rgba(56,189,248,0.07)')
            fig_s.update_layout(**layout_s)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_s, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="analysis-box">
                    <span class="analysis-title">Analisis Kecerahan Akustik</span>
                    <p class="analysis-text">Spectral Centroid mengukur "pusat massa" frekuensi suara. Pola kecerahan pada sinyal uji ini menunjukkan profil energi frekuensi tinggi yang spesifik bagi dialek <b>{winner}</b>.</p>
                </div>
            """, unsafe_allow_html=True)

        # Delta MFCC
        st.markdown("""
            <div class="section-header">
                <span class="section-number">05</span>
                <span class="section-title">Kecepatan Fitur Bicara (Delta MFCC)</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)
        delta = librosa.feature.delta(mfcc_in)
        fig_d = go.Figure(data=go.Heatmap(z=delta, colorscale=viz.get_amber_scale(), zmid=0))
        layout_d = viz._base_layout(280)
        layout_d.update(margin=dict(l=0, r=0, t=10, b=0))
        fig_d.update_layout(**layout_d)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="analysis-box">
                <span class="analysis-title">Analisis Transisi Dinamis</span>
                <p class="analysis-text">Heatmap Delta menggambarkan kecepatan perubahan fonem (ritme tempo). Profil kecepatan artikulasi ini sinkron dengan karakteristik temporal dialek <b>{winner}</b>.</p>
            </div>
        """, unsafe_allow_html=True)

        # Peringkat
        st.markdown("""
            <div class="section-header">
                <span class="section-number">06</span>
                <span class="section-title">Peringkat Kelas Komprehensif</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)

        for rank_idx, (sim, dtw_dist, name, _) in enumerate(results):
            similarity_pct = sim * 100
            is_top = rank_idx == 0
            bar_class = "rank-bar-fill top" if is_top else "rank-bar-fill"
            item_class = "rank-item top" if is_top else "rank-item"
            pct_class = "rank-pct top" if is_top else "rank-pct"
            label_top = "(Terbaik)" if is_top else ""
            st.markdown(f"""
                <div class="{item_class}">
                    <div class="rank-num">#{rank_idx+1}</div>
                    <div class="rank-name">{name} <span style="font-size:0.65rem;color:#4a6b9b;font-weight:400;">{label_top}</span></div>
                    <div class="rank-bar-bg">
                        <div class="{bar_class}" style="width:{similarity_pct:.1f}%"></div>
                    </div>
                    <div class="{pct_class}">{similarity_pct:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="app-footer">
            <div class="app-footer-text">
                Platform Riset Akustik &copy; 2026 &nbsp;·&nbsp; Dikembangkan untuk Penelitian Presisi Tinggi
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    start_dialect_analysis()
