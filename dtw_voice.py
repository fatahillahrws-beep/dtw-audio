import os
import sys
import zipfile
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import streamlit as st
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==============================================================================
# STREAMLIT CONFIG
# ==============================================================================
st.set_page_config(
    page_title="DialectAI · MFCC + GMM",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# UNIVERSAL AUDIO DECODER
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_universal_engine():
    """Menyiapkan decoder agar file mp3/m4a/aac/ogg/flac lebih aman dibaca."""
    try:
        import pydub  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        with st.spinner("Menyiapkan decoder audio universal..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "imageio-ffmpeg"])
            st.rerun()
    return True


initialize_universal_engine()


# ==============================================================================
# MODERN GLASS UI
# ==============================================================================
def apply_modern_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg-0: #030712;
            --bg-1: #07111f;
            --bg-2: #0b1730;
            --glass: rgba(15, 23, 42, 0.62);
            --glass-strong: rgba(15, 23, 42, 0.86);
            --line: rgba(148, 163, 184, 0.18);
            --line-strong: rgba(99, 102, 241, 0.42);
            --text: #f8fafc;
            --sub: #94a3b8;
            --muted: #64748b;
            --cyan: #22d3ee;
            --blue: #60a5fa;
            --violet: #8b5cf6;
            --pink: #ec4899;
            --green: #34d399;
            --amber: #fbbf24;
            --red: #fb7185;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: var(--text);
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 8%, rgba(139, 92, 246, 0.22), transparent 34%),
                radial-gradient(circle at 88% 4%, rgba(34, 211, 238, 0.17), transparent 30%),
                radial-gradient(circle at 58% 110%, rgba(96, 165, 250, 0.16), transparent 35%),
                linear-gradient(135deg, #030712 0%, #08111f 42%, #020617 100%);
        }

        .stApp::before {
            content: '';
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(148, 163, 184, 0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148, 163, 184, 0.035) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(to bottom, rgba(0,0,0,0.8), rgba(0,0,0,0.08));
            z-index: 0;
        }

        .block-container {
            padding-top: 2.2rem !important;
        }

        section[data-testid="stSidebar"] {
            background: rgba(3, 7, 18, 0.88) !important;
            border-right: 1px solid var(--line) !important;
            backdrop-filter: blur(20px);
        }

        section[data-testid="stSidebar"] .block-container {
            padding: 1.5rem 1.1rem !important;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(15,23,42,0.84), rgba(30,41,59,0.50));
            border-radius: 30px;
            padding: 2.8rem 2.8rem;
            box-shadow: 0 30px 90px rgba(0,0,0,0.38);
            backdrop-filter: blur(24px);
            margin-bottom: 2rem;
        }

        .hero::before {
            content: '';
            position: absolute;
            width: 460px;
            height: 460px;
            right: -150px;
            top: -170px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(34,211,238,0.26), transparent 62%);
        }

        .hero::after {
            content: '';
            position: absolute;
            width: 360px;
            height: 360px;
            left: -120px;
            bottom: -160px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(139,92,246,0.24), transparent 64%);
        }

        .hero-content { position: relative; z-index: 2; }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            color: var(--cyan);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            padding: 7px 12px;
            border: 1px solid rgba(34,211,238,0.26);
            border-radius: 999px;
            background: rgba(34,211,238,0.08);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: clamp(2.25rem, 5vw, 4.8rem);
            font-weight: 900;
            letter-spacing: -0.065em;
            line-height: 0.94;
            margin: 0;
        }

        .gradient-text {
            background: linear-gradient(90deg, #22d3ee, #60a5fa, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-subtitle {
            max-width: 760px;
            color: var(--sub);
            font-size: 1rem;
            line-height: 1.75;
            margin-top: 1.1rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 1.4rem;
        }

        .chip {
            color: #dbeafe;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(96,165,250,0.10);
            border: 1px solid rgba(96,165,250,0.22);
        }

        .panel {
            background: linear-gradient(135deg, rgba(15,23,42,0.72), rgba(15,23,42,0.40));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.2rem;
            box-shadow: 0 20px 55px rgba(0,0,0,0.24);
            backdrop-filter: blur(18px);
            margin-bottom: 1rem;
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 1.9rem 0 1rem;
        }

        .section-title .num {
            font-family: 'JetBrains Mono', monospace;
            color: var(--cyan);
            font-size: 0.72rem;
            padding: 5px 9px;
            background: rgba(34,211,238,0.09);
            border: 1px solid rgba(34,211,238,0.20);
            border-radius: 8px;
        }

        .section-title .txt {
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .section-title .line {
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, var(--line), transparent);
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 1rem 0 1.4rem;
        }

        .metric {
            position: relative;
            overflow: hidden;
            padding: 1.15rem 1.05rem;
            min-height: 128px;
            border-radius: 20px;
            border: 1px solid var(--line);
            background: linear-gradient(145deg, rgba(15,23,42,0.86), rgba(30,41,59,0.52));
        }

        .metric::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 82% 12%, rgba(34,211,238,0.12), transparent 36%);
            pointer-events: none;
        }

        .metric-label {
            color: var(--muted);
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            font-size: 0.62rem;
            margin-bottom: 0.85rem;
            position: relative;
        }

        .metric-value {
            font-size: clamp(1.25rem, 2.8vw, 2rem);
            font-weight: 900;
            letter-spacing: -0.04em;
            color: var(--text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            position: relative;
        }

        .metric-sub {
            color: var(--sub);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.66rem;
            margin-top: 0.65rem;
            position: relative;
        }

        .winner-card {
            border-radius: 26px;
            border: 1px solid rgba(34,211,238,0.28);
            padding: 1.6rem;
            background:
                radial-gradient(circle at 88% 20%, rgba(34,211,238,0.15), transparent 32%),
                radial-gradient(circle at 10% 100%, rgba(139,92,246,0.18), transparent 36%),
                linear-gradient(135deg, rgba(15,23,42,0.86), rgba(30,41,59,0.48));
            box-shadow: 0 30px 70px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
        }

        .winner-label {
            color: var(--cyan);
            font-family: 'JetBrains Mono', monospace;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.18em;
        }

        .winner-name {
            font-size: clamp(2.3rem, 6vw, 4.4rem);
            font-weight: 900;
            letter-spacing: -0.07em;
            line-height: 1;
            margin-top: 0.4rem;
        }

        .winner-desc {
            color: var(--sub);
            line-height: 1.65;
            margin-top: 0.8rem;
        }

        .note-box {
            padding: 1rem 1.15rem;
            border-radius: 18px;
            border: 1px solid rgba(148,163,184,0.16);
            background: rgba(15,23,42,0.62);
            color: var(--sub);
            line-height: 1.7;
            margin: 0.8rem 0 1.2rem;
        }

        .rank-item {
            display: grid;
            grid-template-columns: 42px 150px 1fr 82px;
            gap: 12px;
            align-items: center;
            padding: 0.92rem 1rem;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: rgba(15,23,42,0.64);
            margin-bottom: 9px;
        }

        .rank-item.top {
            border-color: rgba(34,211,238,0.36);
            background: linear-gradient(90deg, rgba(34,211,238,0.11), rgba(139,92,246,0.08));
        }

        .rank-num {
            font-family: 'JetBrains Mono', monospace;
            color: var(--muted);
            font-size: 0.78rem;
        }

        .rank-name {
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .bar-bg {
            height: 9px;
            background: rgba(51,65,85,0.7);
            border-radius: 999px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--cyan), var(--blue), var(--violet));
        }

        .rank-pct {
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            color: var(--cyan);
            font-weight: 600;
        }

        .sidebar-logo {
            text-align: center;
            padding: 1.1rem 0.8rem 1.4rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.2rem;
        }

        .sidebar-logo .brand {
            font-size: 1rem;
            font-weight: 900;
            letter-spacing: -0.04em;
        }

        .sidebar-logo .subbrand {
            margin-top: 0.25rem;
            color: var(--muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .sidebar-stat {
            border-radius: 18px;
            padding: 1rem;
            background: linear-gradient(135deg, rgba(34,211,238,0.10), rgba(139,92,246,0.08));
            border: 1px solid rgba(34,211,238,0.18);
            margin-bottom: 1.1rem;
        }

        .sidebar-stat .label {
            color: var(--muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .sidebar-stat .value {
            font-size: 2.25rem;
            font-weight: 900;
            color: var(--cyan);
            line-height: 1;
            margin-top: 0.5rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px !important;
            padding: 9px 18px !important;
            background: rgba(15,23,42,0.54) !important;
            border: 1px solid rgba(148,163,184,0.16) !important;
            color: var(--sub) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.72rem !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
        }

        .stTabs [aria-selected="true"] {
            color: var(--cyan) !important;
            border-color: rgba(34,211,238,0.35) !important;
            background: rgba(34,211,238,0.10) !important;
        }

        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(34,211,238,0.35) !important;
            background: linear-gradient(135deg, rgba(34,211,238,0.18), rgba(139,92,246,0.16)) !important;
            color: #e0f2fe !important;
            font-family: 'JetBrains Mono', monospace !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            font-size: 0.72rem !important;
        }

        .footer {
            text-align: center;
            color: var(--muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            border-top: 1px solid var(--line);
            padding: 1.5rem 0 0.4rem;
            margin-top: 2rem;
        }

        @media (max-width: 1000px) {
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .rank-item { grid-template-columns: 36px 110px 1fr 70px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_modern_styles()


# ==============================================================================
# ACOUSTIC FEATURE CORE: MFCC + DELTA + DELTA-DELTA
# ==============================================================================
class AcousticFeatureCore:
    def __init__(self, sr=16000, n_mfcc=20, max_duration=6.0, top_db=25):
        self.SR = int(sr)
        self.N_MFCC = int(n_mfcc)
        self.MAX_DURATION = float(max_duration)
        self.TOP_DB = int(top_db)

    def load_audio(self, path):
        """Membaca audio memakai pydub lebih dahulu, lalu fallback ke librosa."""
        try:
            import pydub
            import imageio_ffmpeg
            pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            audio = pydub.AudioSegment.from_file(path).set_frame_rate(self.SR).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            denom = float(1 << (8 * audio.sample_width - 1))
            return samples / max(denom, 1.0)
        except Exception:
            y, _ = librosa.load(path, sr=self.SR, mono=True)
            return y.astype(np.float32)

    def clean_audio(self, y):
        """Trim silence, potong durasi, dan normalisasi amplitudo."""
        if y is None or len(y) == 0:
            return None

        max_samples = int(self.SR * self.MAX_DURATION)
        if len(y) > max_samples:
            y = y[:max_samples]

        yt, _ = librosa.effects.trim(y, top_db=self.TOP_DB)
        if len(yt) < int(self.SR * 0.3):
            yt = y

        peak = np.max(np.abs(yt)) if len(yt) > 0 else 0
        if peak > 0:
            yt = yt / (peak + 1e-8)

        return yt.astype(np.float32)

    def extract_frame_features(self, y):
        """
        Menghasilkan fitur frame-level untuk GMM.
        Bentuk output: (T, 3*n_mfcc), terdiri atas MFCC, Delta, Delta-Delta.
        """
        yt = self.clean_audio(y)
        if yt is None or len(yt) == 0:
            return None, None, None

        mfcc = librosa.feature.mfcc(
            y=yt,
            sr=self.SR,
            n_mfcc=self.N_MFCC,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )

        # Normalisasi per koefisien seperti pendekatan dosen: remove mean + scale maksimum absolut.
        mfcc_norm = mfcc.copy()
        for i in range(mfcc_norm.shape[0]):
            mfcc_norm[i] -= np.mean(mfcc_norm[i])
            mx = np.max(np.abs(mfcc_norm[i]))
            if mx > 0:
                mfcc_norm[i] /= (mx + 1e-8)

        delta = librosa.feature.delta(mfcc_norm)
        delta2 = librosa.feature.delta(mfcc_norm, order=2)
        frame_features = np.vstack([mfcc_norm, delta, delta2]).T
        return frame_features.astype(np.float32), mfcc_norm, yt

    def extract_summary_features(self, frames):
        """Ringkasan statistik untuk visualisasi atau model tambahan."""
        return np.concatenate([
            np.mean(frames, axis=0),
            np.std(frames, axis=0),
            np.median(frames, axis=0),
            np.percentile(frames, 25, axis=0),
            np.percentile(frames, 75, axis=0),
        ])


# ==============================================================================
# GMM CLASSIFIER ENGINE
# ==============================================================================
class GMMDialectClassifier:
    def __init__(self, n_components=8, covariance_type="diag", random_state=42, reg_covar=1e-4):
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.reg_covar = float(reg_covar)
        self.scaler = StandardScaler()
        self.models = {}
        self.label_frame_counts = {}
        self.label_file_counts = {}
        self.pca = None
        self.pca_points = None

    def fit(self, frame_bank, file_count_bank):
        """
        frame_bank: dict[label] = list[array(T, F)]
        Setiap dialek dilatih sebagai satu distribusi GMM tersendiri.
        """
        all_frames = []
        for label, items in frame_bank.items():
            for x in items:
                if x is not None and len(x) > 0:
                    all_frames.append(x)

        if not all_frames:
            raise ValueError("Tidak ada fitur audio yang valid untuk melatih GMM.")

        all_frames_stack = np.vstack(all_frames)
        self.scaler.fit(all_frames_stack)

        self.models = {}
        self.label_frame_counts = {}
        self.label_file_counts = dict(file_count_bank)

        for label, items in frame_bank.items():
            frames = np.vstack([x for x in items if x is not None and len(x) > 0])
            frames_scaled = self.scaler.transform(frames)
            n_comp = min(self.n_components, max(1, len(frames_scaled) // 20))
            n_comp = max(1, n_comp)

            model = GaussianMixture(
                n_components=n_comp,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=self.reg_covar,
                max_iter=300,
                n_init=3,
                init_params="kmeans",
            )
            model.fit(frames_scaled)
            self.models[label] = model
            self.label_frame_counts[label] = len(frames_scaled)

        # PCA hanya untuk visualisasi peta fitur database.
        try:
            max_sample = min(6000, len(all_frames_stack))
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(all_frames_stack), size=max_sample, replace=False)
            sample_scaled = self.scaler.transform(all_frames_stack[idx])
            self.pca = PCA(n_components=2, random_state=self.random_state)
            self.pca.fit(sample_scaled)
        except Exception:
            self.pca = None

        return self

    def predict_log_likelihood(self, frames):
        """Menghitung average log-likelihood tiap dialek."""
        x_scaled = self.scaler.transform(frames)
        scores = []
        for label, model in self.models.items():
            avg_ll = float(model.score(x_scaled))
            scores.append((label, avg_ll))
        scores.sort(key=lambda z: z[1], reverse=True)
        return scores

    @staticmethod
    def likelihood_to_probability(scores, temperature=1.0):
        """
        Mengubah log-likelihood menjadi probabilitas relatif dengan softmax.
        Ini bukan probabilitas mutlak, tetapi confidence relatif antar kelas dialek.
        """
        labels = [s[0] for s in scores]
        vals = np.array([s[1] for s in scores], dtype=np.float64)
        vals = vals / max(float(temperature), 1e-6)
        vals = vals - np.max(vals)
        probs = np.exp(vals)
        probs = probs / (np.sum(probs) + 1e-12)
        return list(zip(labels, probs, [s[1] for s in scores]))

    def predict(self, frames, temperature=1.0):
        scores = self.predict_log_likelihood(frames)
        ranked = self.likelihood_to_probability(scores, temperature=temperature)
        ranked.sort(key=lambda z: z[1], reverse=True)
        return ranked


# ==============================================================================
# VISUALIZATION ENGINE
# ==============================================================================
class VizEngine:
    @staticmethod
    def base_layout(height=380, margin=None):
        if margin is None:
            margin = dict(l=10, r=10, t=35, b=10)
        return dict(
            height=height,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
            margin=margin,
        )

    @staticmethod
    def colorscale_cyan_violet():
        return [[0, "#020617"], [0.35, "#1e1b4b"], [0.70, "#2563eb"], [1, "#22d3ee"]]

    @staticmethod
    def colorscale_amber():
        return [[0, "#020617"], [0.45, "#1f2937"], [1, "#fbbf24"]]

    def waveform(self, y):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=y,
            mode="lines",
            line=dict(color="#22d3ee", width=1.3),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.08)",
        ))
        fig.update_layout(**self.base_layout(270))
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.08)")
        return fig

    def mfcc_heatmap(self, mfcc):
        fig = go.Figure(data=go.Heatmap(
            z=mfcc,
            colorscale=self.colorscale_cyan_violet(),
            colorbar=dict(title="MFCC"),
        ))
        fig.update_layout(**self.base_layout(340))
        fig.update_xaxes(title="Frame")
        fig.update_yaxes(title="Koefisien")
        return fig

    def likelihood_bars(self, ranked):
        labels = [x[0] for x in ranked][::-1]
        probs = [x[1] * 100 for x in ranked][::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            marker=dict(
                color=probs,
                colorscale=[[0, "#312e81"], [0.5, "#2563eb"], [1, "#22d3ee"]],
                line=dict(color="rgba(255,255,255,0.14)", width=1),
            ),
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
            hovertemplate="%{y}<br>Confidence relatif: %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(**self.base_layout(360, margin=dict(l=10, r=60, t=30, b=10)))
        fig.update_xaxes(range=[0, max(100, max(probs) * 1.15)], title="Confidence relatif (%)", gridcolor="rgba(148,163,184,0.08)")
        fig.update_yaxes(gridcolor="rgba(148,163,184,0.04)")
        return fig

    def radar(self, ranked):
        labels = [x[0] for x in ranked]
        probs = [x[1] * 100 for x in ranked]
        if len(labels) == 1:
            labels = labels + labels
            probs = probs + probs
        fig = go.Figure(data=go.Scatterpolar(
            r=probs + [probs[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(34,211,238,0.14)",
            line=dict(color="#22d3ee", width=2.5),
            marker=dict(color="#60a5fa", size=6),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(15,23,42,0.35)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.12)"),
                angularaxis=dict(gridcolor="rgba(148,163,184,0.10)"),
            ),
            **self.base_layout(360),
        )
        return fig

    def spectral_centroid(self, y, sr):
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=sc,
            line=dict(color="#fbbf24", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(251,191,36,0.08)",
            name="Spectral Centroid",
        ))
        fig.update_layout(**self.base_layout(300))
        fig.update_xaxes(title="Frame", gridcolor="rgba(148,163,184,0.08)")
        fig.update_yaxes(title="Hz", gridcolor="rgba(148,163,184,0.08)")
        return fig

    def pca_projection(self, classifier, frames, label_pred):
        if classifier.pca is None:
            return None
        scaled = classifier.scaler.transform(frames)
        pts = classifier.pca.transform(scaled)
        max_pts = min(len(pts), 900)
        pts = pts[:max_pts]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode="markers",
            marker=dict(size=6, color="#22d3ee", opacity=0.68, line=dict(width=0)),
            name=f"Audio uji → {label_pred}",
        ))
        fig.update_layout(**self.base_layout(300))
        fig.update_xaxes(title="PC1", gridcolor="rgba(148,163,184,0.08)")
        fig.update_yaxes(title="PC2", gridcolor="rgba(148,163,184,0.08)")
        return fig


# ==============================================================================
# DATABASE BOOTSTRAP
# ==============================================================================
@st.cache_resource(show_spinner=False)
def boot_database(n_components, covariance_type, sr, n_mfcc, max_duration, top_db):
    core = AcousticFeatureCore(sr=sr, n_mfcc=n_mfcc, max_duration=max_duration, top_db=top_db)
    zip_files = sorted(list(Path(".").glob("*.zip")))

    frame_bank = defaultdict(list)
    wave_bank = defaultdict(list)
    file_count_bank = defaultdict(int)

    for z in zip_files:
        label = z.stem.replace("Logat_", "").replace("logat_", "").upper()
        with tempfile.TemporaryDirectory() as td:
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(td)
            except Exception:
                continue

            for f in Path(td).rglob("*"):
                if f.suffix.lower() in [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]:
                    try:
                        y = core.load_audio(str(f))
                        frames, mfcc, yt = core.extract_frame_features(y)
                        if frames is not None and len(frames) > 5:
                            frame_bank[label].append(frames)
                            wave_bank[label].append(yt)
                            file_count_bank[label] += 1
                    except Exception:
                        continue

    if not frame_bank:
        return None, None, None, [], {}

    clf = GMMDialectClassifier(n_components=n_components, covariance_type=covariance_type)
    clf.fit(frame_bank, file_count_bank)
    return clf, core, wave_bank, zip_files, dict(file_count_bank)


# ==============================================================================
# SMALL UI HELPERS
# ==============================================================================
def section(num, title):
    st.markdown(
        f"""
        <div class="section-title">
            <span class="num">{num}</span>
            <span class="txt">{title}</span>
            <span class="line"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rank(ranked):
    for i, (label, prob, ll) in enumerate(ranked, start=1):
        pct = prob * 100
        top_class = " top" if i == 1 else ""
        st.markdown(
            f"""
            <div class="rank-item{top_class}">
                <div class="rank-num">#{i}</div>
                <div class="rank-name">{label}</div>
                <div class="bar-bg"><div class="bar-fill" style="width:{pct:.2f}%"></div></div>
                <div class="rank-pct">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# MAIN APP
# ==============================================================================
def start_gmm_dialect_app():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="brand"><span class="gradient-text">DialectAI</span></div>
                <div class="subbrand">MFCC · GMM Classifier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        zip_count = len(list(Path(".").glob("*.zip")))
        st.markdown(
            f"""
            <div class="sidebar-stat">
                <div class="label">Database Terdeteksi</div>
                <div class="value">{zip_count}</div>
                <div style="color:#94a3b8;font-size:0.78rem;margin-top:0.35rem;">arsip logat .zip</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Parameter Model**")
        n_components = st.slider("Jumlah komponen GMM", 1, 24, 8, help="Komponen Gaussian. Naikkan jika data per dialek banyak.")
        covariance_type = st.selectbox("Tipe kovarians", ["diag", "full", "tied", "spherical"], index=0)
        temperature = st.slider("Kalibrasi confidence", 0.3, 3.0, 1.0, 0.1, help="Lebih kecil = confidence lebih tajam; lebih besar = lebih halus.")

        st.markdown("**Parameter Audio**")
        sr = st.selectbox("Sampling rate", [16000, 22050], index=0)
        n_mfcc = st.slider("Jumlah MFCC", 13, 40, 20)
        max_duration = st.slider("Durasi maksimum audio", 3.0, 12.0, 6.0, 0.5)
        top_db = st.slider("Trim silence top_db", 15, 45, 25)

        if st.button("Muat Ulang Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.markdown(
            """
            <div class="note-box" style="font-size:0.78rem;">
            Model ini melatih satu GMM untuk setiap dialek. Audio uji dipilih berdasarkan average log-likelihood tertinggi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="eyebrow">🎙️ Probabilistic Acoustic Intelligence</div>
                <h1 class="hero-title">Dialect Recognition<br><span class="gradient-text">MFCC + GMM</span></h1>
                <div class="hero-subtitle">
                    Sistem klasifikasi dialek berbasis fitur MFCC, Delta, dan Delta-Delta. Setiap dialek dimodelkan sebagai distribusi campuran Gaussian sehingga kemiripan dihitung berdasarkan likelihood statistik, bukan penyelarasan temporal DTW.
                </div>
                <div class="chip-row">
                    <span class="chip">MFCC Frame-Level</span>
                    <span class="chip">Gaussian Mixture Model</span>
                    <span class="chip">Log-Likelihood Scoring</span>
                    <span class="chip">Softmax Confidence</span>
                    <span class="chip">Modern Visual Analytics</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    clf, core, wave_bank, zip_files, file_counts = boot_database(
        n_components=n_components,
        covariance_type=covariance_type,
        sr=sr,
        n_mfcc=n_mfcc,
        max_duration=max_duration,
        top_db=top_db,
    )

    if clf is None:
        st.error("Database akustik tidak ditemukan. Letakkan file .zip logat di direktori yang sama dengan script ini, misalnya Logat_BUGIS.zip, Logat_JAWA.zip, dan seterusnya.")
        return

    total_files = sum(file_counts.values())
    total_frames = sum(clf.label_frame_counts.values())
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric"><div class="metric-label">Kelas Dialek</div><div class="metric-value gradient-text">{len(clf.models)}</div><div class="metric-sub">model GMM aktif</div></div>
            <div class="metric"><div class="metric-label">Sampel Audio</div><div class="metric-value">{total_files}</div><div class="metric-sub">file referensi</div></div>
            <div class="metric"><div class="metric-label">Frame Fitur</div><div class="metric-value">{total_frames:,}</div><div class="metric-sub">MFCC + Δ + ΔΔ</div></div>
            <div class="metric"><div class="metric-label">Dimensi Fitur</div><div class="metric-value">{n_mfcc*3}</div><div class="metric-sub">per frame audio</div></div>
        </div>
        """.replace(",", "."),
        unsafe_allow_html=True,
    )

    section("INPUT", "Pilih atau Rekam Audio Uji")
    tab_upload, tab_record = st.tabs(["Unggah Audio", "Rekam Langsung"])
    audio_stream, source_id = None, ""

    with tab_upload:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Unggah file audio",
            type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
            help="Format yang didukung: WAV, MP3, M4A, AAC, FLAC, OGG.",
        )
        if uploaded:
            audio_stream = uploaded.read()
            source_id = uploaded.name
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_record:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        recorded = st.audio_input("Rekam suara untuk diuji")
        if recorded:
            audio_stream = recorded.read()
            source_id = "rekaman_langsung.wav"
        st.markdown('</div>', unsafe_allow_html=True)

    viz = VizEngine()

    if audio_stream:
        with st.spinner("Mengekstraksi MFCC dan menghitung likelihood GMM..."):
            suffix = Path(source_id).suffix if Path(source_id).suffix else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_stream)
                tmp_path = tmp.name

            try:
                y_raw = core.load_audio(tmp_path)
                frames, mfcc_norm, y_trim = core.extract_frame_features(y_raw)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            if frames is None or len(frames) < 5:
                st.error("Fitur audio gagal diekstraksi. Coba gunakan rekaman yang lebih jelas dan berdurasi minimal 1 detik.")
                return

            ranked = clf.predict(frames, temperature=temperature)
            winner, confidence, winner_ll = ranked[0]

        section("HASIL", "Prediksi Dialek Berbasis Likelihood")
        st.markdown(
            f"""
            <div class="winner-card">
                <div class="winner-label">Prediksi Dialek Terkuat</div>
                <div class="winner-name"><span class="gradient-text">{winner}</span></div>
                <div class="winner-desc">
                    Audio uji paling sesuai dengan model distribusi dialek <b>{winner}</b>. Confidence relatif dihitung dari softmax average log-likelihood seluruh model GMM.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        duration = len(y_trim) / core.SR
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric"><div class="metric-label">Confidence</div><div class="metric-value gradient-text">{confidence*100:.1f}%</div><div class="metric-sub">softmax likelihood</div></div>
                <div class="metric"><div class="metric-label">Log-Likelihood</div><div class="metric-value">{winner_ll:.2f}</div><div class="metric-sub">average per frame</div></div>
                <div class="metric"><div class="metric-label">Durasi Bersih</div><div class="metric-value">{duration:.2f}s</div><div class="metric-sub">setelah trim silence</div></div>
                <div class="metric"><div class="metric-label">Frame Uji</div><div class="metric-value">{len(frames)}</div><div class="metric-sub">fitur dianalisis</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section("01", "Peringkat Similarity Antar Dialek")
        col1, col2 = st.columns([1.25, 0.9])
        with col1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(viz.likelihood_bars(ranked), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            render_rank(ranked)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="note-box">
            Pada pendekatan GMM, similarity tidak dihitung dari jarak lintasan seperti DTW. Similarity dihitung dari seberapa besar peluang fitur frame-level audio uji muncul pada distribusi Gaussian dialek tertentu. Dialek <b>{winner}</b> memperoleh likelihood tertinggi sehingga menjadi kelas prediksi utama.
            </div>
            """,
            unsafe_allow_html=True,
        )

        section("02", "Visualisasi Sinyal dan MFCC")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(viz.waveform(y_trim), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(viz.mfcc_heatmap(mfcc_norm), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        section("03", "Radar Confidence dan Kecerahan Akustik")
        col5, col6 = st.columns(2)
        with col5:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(viz.radar(ranked), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col6:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(viz.spectral_centroid(y_trim, core.SR), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        section("04", "Peta Fitur Audio Uji")
        pca_fig = viz.pca_projection(clf, frames, winner)
        if pca_fig is not None:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.plotly_chart(pca_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("PCA tidak tersedia karena jumlah fitur database belum cukup.")

    else:
        section("STATUS", "Model Siap Digunakan")
        st.markdown(
            """
            <div class="note-box">
            Unggah atau rekam audio untuk memulai klasifikasi. Pastikan rekaman cukup jelas, tidak terlalu banyak noise, dan berdurasi minimal sekitar satu detik agar fitur MFCC dapat terbaca stabil.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            DialectAI · MFCC + Gaussian Mixture Model · Probabilistic Acoustic Classification
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    start_gmm_dialect_app()
