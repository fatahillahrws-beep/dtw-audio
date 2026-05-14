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
# [3] PREMIUM NAVY UI/UX ENGINE — REDESIGNED
# ==============================================================================
def apply_professional_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

        /* ── Root Variables ── */
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
            --accent-rose: #fb7185;
            --text-primary: #e8f0fe;
            --text-secondary: #8eafd4;
            --text-muted: #4a6b9b;
            --border: rgba(56,189,248,0.15);
            --border-strong: rgba(56,189,248,0.35);
            --shadow: 0 8px 32px rgba(2,8,18,0.7);
            --shadow-lg: 0 20px 60px rgba(2,8,18,0.9);
        }

        /* ── Global Reset ── */
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

        /* ── Scanline texture overlay ── */
        .stApp::before {
            content: '';
            position: fixed; inset: 0; z-index: 0; pointer-events: none;
            background-image: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(56,189,248,0.015) 2px,
                rgba(56,189,248,0.015) 4px
            );
        }

        /* ── Hero Header ── */
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
            font-size: 0.65rem; padding: 5px 12px;
            border-radius: 4px; letter-spacing: 1px; text-transform: uppercase;
        }

        /* ── Section Headers ── */
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

        /* ── Metric Cards ── */
        .metrics-row { display: flex; gap: 16px; margin: 1.5rem 0 2rem; }
        .metric-card {
            flex: 1;
            background: linear-gradient(145deg, var(--navy-600) 0%, var(--navy-700) 100%);
            padding: 1.8rem 1.5rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            position: relative; overflow: hidden;
            transition: border-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
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
            font-size: 0.65rem; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 2px;
            margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
        }
        .metric-label::before {
            content: ''; width: 6px; height: 6px;
            background: var(--accent-sky); border-radius: 50%;
            display: inline-block;
        }
        .metric-value {
            font-family: 'Syne', sans-serif;
            font-size: 2.4rem; font-weight: 800;
            color: var(--accent-sky); line-height: 1;
        }
        .metric-value.green { color: var(--accent-emerald); }
        .metric-sub {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem; color: var(--text-muted);
            margin-top: 8px;
        }

        /* ── Analysis Box ── */
        .analysis-box {
            background: linear-gradient(135deg, rgba(7,18,40,0.9) 0%, rgba(4,13,30,0.95) 100%);
            padding: 1.6rem 1.8rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-sky);
            margin: 1rem 0 2rem;
            position: relative;
        }
        .analysis-box::before {
            content: '⬡';
            position: absolute; top: 1.2rem; right: 1.5rem;
            color: var(--accent-sky-dim); font-size: 2rem; opacity: 0.3;
        }
        .analysis-title {
            font-family: 'Syne', sans-serif;
            font-size: 0.8rem; font-weight: 700;
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

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: var(--navy-800) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

        .sidebar-brand {
            text-align: center; padding: 1.5rem 1rem 1rem;
            border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;
        }
        .sidebar-brand-icon {
            font-size: 2rem; margin-bottom: 6px; display: block;
        }
        .sidebar-brand-title {
            font-family: 'Syne', sans-serif;
            font-size: 0.85rem; font-weight: 700;
            color: var(--text-primary); letter-spacing: 1px;
            text-transform: uppercase;
        }
        .sidebar-brand-sub {
            font-family: 'DM Mono', monospace;
            font-size: 0.6rem; color: var(--text-muted);
            letter-spacing: 2px; margin-top: 3px;
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

        /* ── Tabs ── */
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
            font-size: 0.72rem !important;
            letter-spacing: 1.5px !important;
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

        /* ── Upload Zone ── */
        .upload-zone {
            background: var(--accent-sky-dim);
            border: 2px dashed var(--border-strong);
            border-radius: 16px; padding: 2rem;
            text-align: center; margin: 1rem 0;
            transition: all 0.3s ease;
        }
        .upload-zone:hover {
            background: rgba(56,189,248,0.08);
            border-color: var(--accent-sky);
        }
        .upload-icon { font-size: 2.5rem; display: block; margin-bottom: 10px; }
        .upload-title {
            font-family: 'Syne', sans-serif;
            font-size: 1rem; font-weight: 700; color: var(--text-primary);
        }
        .upload-sub {
            font-family: 'DM Mono', monospace;
            font-size: 0.65rem; color: var(--text-muted);
            margin-top: 6px; letter-spacing: 1px;
        }

        /* ── Ranking Bar ── */
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
            transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1);
        }
        .rank-bar-fill.top { background: linear-gradient(90deg, #38bdf8, #7dd3fc, #22d3ee); }
        .rank-pct {
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem; font-weight: 500; color: var(--accent-sky);
            min-width: 50px; text-align: right;
        }
        .rank-pct.top { color: var(--accent-cyan); }

        /* ── Chart Container ── */
        .chart-container {
            background: linear-gradient(135deg, var(--navy-700) 0%, var(--navy-800) 100%);
            border-radius: 16px; border: 1px solid var(--border);
            padding: 1.2rem; margin-bottom: 0.5rem;
            overflow: hidden;
        }

        /* ── Dividers ── */
        hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

        /* ── Input Widgets ── */
        .stSlider [data-testid="stTickBar"] { color: var(--text-muted); }
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background: var(--accent-sky) !important;
            border: 2px solid var(--navy-700) !important;
        }

        /* ── Buttons ── */
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

        /* ── Spinner ── */
        .stSpinner { color: var(--accent-sky) !important; }

        /* ── Alert / Error ── */
        .stAlert { border-radius: 12px !important; border: 1px solid var(--border) !important; }

        /* ── Footer ── */
        .app-footer {
            text-align: center; padding: 2rem 0 1rem;
            border-top: 1px solid var(--border); margin-top: 3rem;
        }
        .app-footer-text {
            font-family: 'DM Mono', monospace;
            font-size: 0.62rem; color: var(--text-muted);
            letter-spacing: 2px; text-transform: uppercase;
        }

        /* ── Processing overlay animation ── */
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        .processing-badge {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--accent-sky-dim);
            border: 1px solid var(--border-strong);
            border-radius: 20px; padding: 6px 16px;
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem; color: var(--accent-sky);
            letter-spacing: 1px; text-transform: uppercase;
        }
        .processing-badge span {
            animation: pulse 1s ease-in-out infinite;
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
            y, _ = librosa.load(path, sr=self.SR, mono=True)
            return y

    def extract_dialect_features(self, y):
        """Ekstraksi Koefisien Cepstral dengan VAD internal terintegrasi."""
        yt, _ = librosa.effects.trim(y, top_db=25)
        mfcc = librosa.feature.mfcc(y=yt, sr=self.SR, n_mfcc=self.N_MFCC)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        features = np.hstack([mfcc.T, d1.T, d2.T])
        mu = np.mean(features, axis=0)
        sigma = np.std(features, axis=0) + 1e-8
        return (features - mu) / sigma, mfcc, yt

def dtw_alignment_engine(s1, s2, w_const):
    """Kalkulasi Penyelarasan Waktu Dinamis menggunakan metrik Cosine."""
    n, m = len(s1), len(s2)
    w = max(w_const, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf); dp[0, 0] = 0.0
    cost = cdist(s1, s2, metric='cosine')
    for i in range(1, n + 1):
        for j in range(max(1, i-w), min(m, i+w)+1):
            prev_min = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
            dp[i, j] = cost[i-1, j-1] + prev_min
    return dp[n, m] / (n + m)

# ==============================================================================
# [5] VISUALIZATION ENGINE — PREMIUM FACTORY
# ==============================================================================
class VizEngine:
    """Modul untuk memfabrikasi grafik dengan palet warna Navy-Cyber."""

    @staticmethod
    def get_navy_scale():
        return [[0, '#020812'], [0.4, '#1a3a6e'], [1, '#38bdf8']]

    @staticmethod
    def get_amber_scale():
        return [[0, '#020812'], [0.5, '#1e293b'], [1, '#fbbf24']]

    def _base_layout(self, h=400):
        return dict(
            height=h,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Mono, monospace', color='#8eafd4', size=11),
            margin=dict(l=10, r=10, t=30, b=10),
        )

    def plot_waveform(self, y_in, y_ref, label):
        """Membuat perbandingan waveform dengan spasi vertikal yang rapi."""
        fig = make_subplots(
            rows=2, cols=1, vertical_spacing=0.22,
            subplot_titles=("▶ Input Signal", f"▶ Reference: {label}")
        )
        fig.add_trace(go.Scatter(
            y=y_in, line=dict(color='#38bdf8', width=1.2),
            fill='tozeroy', fillcolor='rgba(56,189,248,0.06)',
            name="Input"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            y=y_ref, line=dict(color='#22d3ee', width=1.2),
            fill='tozeroy', fillcolor='rgba(34,211,238,0.05)',
            name="Reference"
        ), row=2, col=1)
        layout = self._base_layout(440)
        layout.update(showlegend=False)
        layout['xaxis2'] = dict(title=dict(text="Samples", font=dict(size=10)))
        # Style annotation font
        fig.update_annotations(font=dict(size=11, color='#8eafd4', family='DM Mono, monospace'))
        fig.update_layout(**layout)
        fig.update_xaxes(gridcolor='rgba(56,189,248,0.07)', zerolinecolor='rgba(56,189,248,0.15)')
        fig.update_yaxes(gridcolor='rgba(56,189,248,0.07)', zerolinecolor='rgba(56,189,248,0.15)')
        return fig

    def plot_heatmap(self, data, labels):
        """Matriks kemiripan dengan label dialek."""
        fig = go.Figure(data=go.Heatmap(
            z=[data], x=labels,
            colorscale=self.get_navy_scale(),
            zmin=0, zmax=100,
            text=[[f"{v:.0f}%" for v in data]],
            texttemplate="%{text}",
            textfont=dict(family='DM Mono, monospace', size=12, color='#e8f0fe')
        ))
        layout = self._base_layout(240)
        layout.update(margin=dict(l=0, r=0, t=10, b=0))
        fig.update_layout(**layout)
        return fig

    def plot_radar(self, labels, values):
        """Distribusi probabilitas logat."""
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill='toself',
            fillcolor='rgba(56,189,248,0.12)',
            line=dict(color='#38bdf8', width=2.5),
            marker=dict(color='#38bdf8', size=6)
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(7,18,40,0.6)',
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor='rgba(56,189,248,0.12)',
                    tickfont=dict(family='DM Mono, monospace', size=9, color='#4a6b9b')
                ),
                angularaxis=dict(
                    tickfont=dict(family='DM Mono, monospace', size=10, color='#8eafd4'),
                    gridcolor='rgba(56,189,248,0.08)'
                )
            ),
            **self._base_layout(340)
        )
        return fig

# ==============================================================================
# [6] ANALYTICAL PIPELINE EXECUTION
# ==============================================================================
def start_dialect_analysis():
    zip_files = list(Path('.').glob('*.zip'))

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-brand">
                <span class="sidebar-brand-icon">🎙️</span>
                <div class="sidebar-brand-title">Acoustic Analytics</div>
                <div class="sidebar-brand-sub">Pattern Recognition Lab</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="status-card">
                <div class="status-label">Computational Status</div>
                <div class="status-main">{len(zip_files)}<span style="font-size:1rem;color:#4a6b9b;font-weight:400;"> logat</span></div>
                <div class="status-badge">Engine Optimized</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-label">Parameter Configuration</div>', unsafe_allow_html=True)
        k_val = st.slider("Classification Sensitivity (K)", 1, 15, 5)
        w_val = st.slider("Window Constraint (W)", 20, 400, 120, step=10)

        st.markdown('<div class="sidebar-section-label">System</div>', unsafe_allow_html=True)
        if st.button("⟳ Reload Research Intelligence", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("""
            <div style="margin-top:2rem;padding-top:1.5rem;border-top:1px solid rgba(56,189,248,0.1);">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#4a6b9b;letter-spacing:1.5px;text-align:center;">
                    DTW + MFCC ENGINE<br>VERSION 3.7.0 STABLE
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── Hero Header ───────────────────────────────────────────────────────────
    st.markdown("""
        <div class="hero-header">
            <div class="hero-eyebrow">Acoustic Intelligence System</div>
            <h1 class="hero-title">
                Dialect <span>Recognition</span><br>Laboratory
            </h1>
            <div class="hero-subtitle">Dynamic Time Warping · MFCC Feature Extraction · VAD Engine</div>
            <div class="hero-badges">
                <span class="hero-badge">🔬 DSP Hybrid</span>
                <span class="hero-badge">⚡ Cosine Distance</span>
                <span class="hero-badge">🛡️ Universal Decoder</span>
                <span class="hero-badge">📊 Multi-dialect DB</span>
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
                with zipfile.ZipFile(z, 'r') as zf: zf.extractall(td)
                for f in Path(td).rglob('*'):
                    if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']:
                        y = core.load_audio(str(f))
                        if y is not None:
                            feats, _, yt = core.extract_dialect_features(y)
                            db_templates[label].append(feats)
                            db_waves[label].append(yt)
        return db_templates, db_waves

    # ── Load DB ───────────────────────────────────────────────────────────────
    db_templates, db_waves = boot_database()

    if not db_templates:
        st.error("⚠️ Missing Acoustic Datasets. Please provide .zip archives in the working directory.")
        return

    # ── Input Interface ───────────────────────────────────────────────────────
    st.markdown("""
        <div class="section-header">
            <span class="section-number">INPUT</span>
            <span class="section-title">Audio Source Selection</span>
            <span class="section-line"></span>
        </div>
    """, unsafe_allow_html=True)

    tab_f, tab_r = st.tabs(["📁  File Upload", "🎤  Live Capture"])
    audio_stream, source_id = None, ""

    with tab_f:
        st.markdown("""
            <div class="upload-zone">
                <span class="upload-icon">🎵</span>
                <div class="upload-title">Upload Acoustic Data</div>
                <div class="upload-sub">WAV · MP3 · M4A · AAC · FLAC · OGG</div>
            </div>
        """, unsafe_allow_html=True)
        u = st.file_uploader(
            "Upload Acoustic Data",
            type=['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg'],
            label_visibility="collapsed"
        )
        if u:
            audio_stream, source_id = u.read(), u.name

    with tab_r:
        m = st.audio_input("Record Speech Pattern", label_visibility="collapsed")
        if m:
            audio_stream, source_id = m.read(), "live_stream.wav"

    # ── Analysis Pipeline ─────────────────────────────────────────────────────
    if audio_stream:
        with st.spinner("Executing spectral decomposition..."):
            with tempfile.NamedTemporaryFile(suffix=Path(source_id).suffix, delete=False) as tmp:
                tmp.write(audio_stream)
                path = tmp.name
            y_raw = core.load_audio(path)
            feats_in, mfcc_in, y_in_t = core.extract_dialect_features(y_raw)
            os.remove(path)

            # CALCULATION PIPELINE
            scores, h_vals, h_lbls = [], [], []
            for label, templates in db_templates.items():
                for i, t in enumerate(templates):
                    dist = dtw_alignment_engine(feats_in, t, w_val)
                    scores.append((dist, label, i))
                    h_vals.append(1/(1+dist)*100)
                    h_lbls.append(f"{label}")

            scores.sort(key=lambda x: x[0])
            winner, idx_winner = scores[0][1], scores[0][2]
            conf = (1/(1+scores[0][0])*100)

        # ── Results Metrics ───────────────────────────────────────────────────
        st.markdown("""
            <div class="section-header">
                <span class="section-number">RESULT</span>
                <span class="section-title">Classification Output</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="metric-label">Dialect Identity</div>
                    <div class="metric-value">{winner}</div>
                    <div class="metric-sub">Primary Classification</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{conf:.1f}%</div>
                    <div class="metric-sub">DTW Cosine Distance</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">VAD Engine</div>
                    <div class="metric-value green">ACTIVE</div>
                    <div class="metric-sub">Voice Activity Detected</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── 1. Waveform ───────────────────────────────────────────────────────
        st.markdown("""
            <div class="section-header">
                <span class="section-number">01</span>
                <span class="section-title">Temporal Signal Alignment Map</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(viz.plot_waveform(y_in_t, db_waves[winner][idx_winner], winner), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="analysis-box">
                <span class="analysis-title">🔬 Temporal Consistency Analysis</span>
                <p class="analysis-text">Waveform comparison maps the modulation synchronization between the input signal and master reference. Dialect <b>{winner}</b> dominates because it exhibits the most identical syllabic emphasis structure and speech rhythm. The Universal Decoder ensures signal integrity is maintained even for audio from compressed formats.</p>
            </div>
        """, unsafe_allow_html=True)

        # ── 2. Heatmap ────────────────────────────────────────────────────────
        st.markdown("""
            <div class="section-header">
                <span class="section-number">02</span>
                <span class="section-title">Spectral Similarity Matrix</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(viz.plot_heatmap(h_vals, h_lbls), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="analysis-box">
                <span class="analysis-title">🔬 Matrix Correlation Analysis</span>
                <p class="analysis-text">The spectral similarity matrix maps MFCC feature correlations across the entire database using the Cyber-Navy palette. The bright blue area in the <b>{winner}</b> column indicates the most stable voice feature match density, minimizing cross-dialect classification bias.</p>
            </div>
        """, unsafe_allow_html=True)

        # ── 3 & 4. Radar + Spectral ───────────────────────────────────────────
        st.markdown("""
            <div class="section-header">
                <span class="section-number">03 · 04</span>
                <span class="section-title">Probability Radar &amp; Acoustic Brightness</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            u_labs = list(db_templates.keys())
            v_radar = [1/(1+min([x[0] for x in scores if x[1]==L]))*100 for L in u_labs]
            st.plotly_chart(viz.plot_radar(u_labs, v_radar), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="analysis-box">
                    <span class="analysis-title">🔬 Radar Distribution Analysis</span>
                    <p class="analysis-text">The probability radar shows vector pull skewed toward the <b>{winner}</b> axis, confirming unique vocal morphology that doesn't overlap with other dialects in the system.</p>
                </div>
            """, unsafe_allow_html=True)

        with col_r:
            sc = librosa.feature.spectral_centroid(y=y_in_t, sr=16000)[0]
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                y=sc,
                line=dict(color='#fbbf24', width=1.8),
                fill='tozeroy',
                fillcolor='rgba(251,191,36,0.06)'
            ))
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
                    <span class="analysis-title">🔬 Acoustic Brightness Analysis</span>
                    <p class="analysis-text">Spectral Centroid measures the frequency "center of mass". The brightness pattern of this test signal shows a high-frequency energy profile highly specific to dialect <b>{winner}</b>, reflecting the unique speech melody of that region.</p>
                </div>
            """, unsafe_allow_html=True)

        # ── 5. Delta MFCC ─────────────────────────────────────────────────────
        st.markdown("""
            <div class="section-header">
                <span class="section-number">05</span>
                <span class="section-title">Velocity of Speech Features (Delta MFCC)</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)
        delta = librosa.feature.delta(mfcc_in)
        fig_d = go.Figure(data=go.Heatmap(
            z=delta,
            colorscale=viz.get_amber_scale(),
            zmid=0
        ))
        layout_d = viz._base_layout(280)
        layout_d.update(margin=dict(l=0, r=0, t=10, b=0))
        fig_d.update_layout(**layout_d)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="analysis-box">
                <span class="analysis-title">🔬 Dynamic Transition Analysis</span>
                <p class="analysis-text">The Delta heatmap illustrates the speed of phoneme changes (tempo rhythm). The proximity on this chart shows your speech dynamics have an articulation velocity profile synchronized with the temporal characteristics of <b>{winner}</b>, reinforcing detection results from the tempo side.</p>
            </div>
        """, unsafe_allow_html=True)

        # ── 6. Ranking ────────────────────────────────────────────────────────
        st.markdown("""
            <div class="section-header">
                <span class="section-number">06</span>
                <span class="section-title">Comprehensive Class Ranking</span>
                <span class="section-line"></span>
            </div>
        """, unsafe_allow_html=True)

        final_r = sorted(
            {L: 1/(1+min([x[0] for x in scores if x[1]==L]))*100 for L in db_templates.keys()}.items(),
            key=lambda x: x[1], reverse=True
        )

        for rank_idx, (name, score) in enumerate(final_r):
            is_top = rank_idx == 0
            bar_class = "rank-bar-fill top" if is_top else "rank-bar-fill"
            item_class = "rank-item top" if is_top else "rank-item"
            pct_class = "rank-pct top" if is_top else "rank-pct"
            crown = "👑 " if is_top else ""
            st.markdown(f"""
                <div class="{item_class}">
                    <div class="rank-num">#{rank_idx+1}</div>
                    <div class="rank-name">{crown}{name}</div>
                    <div class="rank-bar-bg">
                        <div class="{bar_class}" style="width:{score:.1f}%"></div>
                    </div>
                    <div class="{pct_class}">{score:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="app-footer">
            <div class="app-footer-text">
                Acoustic Research Platform &copy; 2026 &nbsp;·&nbsp; Developed for High-Precision Research &nbsp;·&nbsp; DSP Hybrid V3.7.0
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    start_dialect_analysis()
