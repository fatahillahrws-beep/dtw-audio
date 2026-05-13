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
    page_title="AI Dialect Analysis System",
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
# PROFESSIONAL CSS STYLING
# ============================================================
st.markdown("""
<style>
    /* Dark Theme Optimization */
    .stApp { background-color: #0e1117; }
    
    /* Header Professional */
    .main-header {
        background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 5px solid #58a6ff;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .main-header h1 {
        color: #f0f6fc;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .main-header p {
        color: #8b949e;
        margin-top: 0.5rem;
        font-size: 1rem;
    }

    /* Professional Metrics Card */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        flex: 1;
        background: #161b22;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #30363d;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #58a6ff;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border: none;
        color: #8b949e;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #58a6ff; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

    /* Info Box */
    .status-box {
        padding: 1rem;
        background: #161b22;
        border-radius: 6px;
        border-left: 3px solid #238636;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CLASSIFIER LOGIC (REVISED)
# ============================================================
class AudioConfig:
    def __init__(self):
        self.SAMPLE_RATE = 16000
        self.N_MFCC = 13
        self.N_MELS = 40
        self.HOP_LENGTH = 160
        self.WIN_LENGTH = 400
        self.N_FFT = 512
        self.DELTA = True
        self.DELTA_DELTA = True
        self.MIN_DURATION = 0.1
        self.FIXED_DURATION = None

def load_audio_safely(filepath, sr=16000):
    try:
        import pydub
        import imageio_ffmpeg
        pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        audio = pydub.AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        return samples / (2**15)
    except:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        return y

class FeatureExtractor:
    def __init__(self, config):
        self.cfg = config
        self.scaler = StandardScaler()

    def extract_mfcc(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=self.cfg.SAMPLE_RATE, n_mfcc=self.cfg.N_MFCC)
        feat = [mfcc.T]
        if self.cfg.DELTA: feat.append(librosa.feature.delta(mfcc).T)
        if self.cfg.DELTA_DELTA: feat.append(librosa.feature.delta(mfcc, order=2).T)
        return np.hstack(feat)

    def process(self, bytes_data, name):
        with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as tmp:
            tmp.write(bytes_data)
            path = tmp.name
        try:
            y = load_audio_safely(path, self.cfg.SAMPLE_RATE)
            y, _ = librosa.effects.trim(y, top_db=30)
            if len(y) < 1600: return None, None
            return self.extract_mfcc(y), y
        finally: os.remove(path)

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # HEADER
    st.markdown("""
        <div class="main-header">
            <h1>Acoustic Dialect Fingerprinting</h1>
            <p>Advanced Speech Pattern Matching using Dynamic Time Warping & Mel-Frequency Cepstral Coefficients</p>
        </div>
    """, unsafe_allow_html=True)

    # SIDEBAR CONFIGURATION
    with st.sidebar:
        st.markdown("### Model Configuration")
        
        # K-Neighbors Control
        k_val = st.slider("K-Nearest Neighbors", 1, 15, 5, 
                          help="Jumlah sampel referensi terdekat yang dipertimbangkan untuk klasifikasi.")
        
        # DTW Window Control
        st.markdown("---")
        st.markdown("### DTW Constraints")
        dtw_w = st.slider("Sakoe-Chiba Window", 10, 200, 50, step=10,
                         help="Batasan pencarian sekuens waktu. Nilai lebih kecil membuat komputasi lebih cepat namun kurang fleksibel.")
        st.caption("Unit: Time Frames (1 frame ≈ 10ms)")
        
        st.markdown("---")
        st.markdown("### Data Management")
        if st.button("Refresh Training Database", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # INITIALIZE ENGINE
    cfg = AudioConfig()
    extractor = FeatureExtractor(cfg)
    
    # LOAD DATABASE (Mock logic based on previous ZIP requirements)
    @st.cache_resource
    def load_db():
        templates = defaultdict(list)
        raw_waves = defaultdict(list)
        zip_files = list(Path('.').glob('*.zip'))
        
        if not zip_files: return None, "No ZIP data found."
        
        for zf in zip_files:
            cname = zf.stem.replace("Logat_", "")
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zf, 'r') as z: z.extractall(tmpdir)
                for p in Path(tmpdir).rglob('*'):
                    if p.suffix.lower() in ['.wav', '.mp3', '.m4a']:
                        y = load_audio_safely(str(p))
                        if y is not None:
                            templates[cname].append(extractor.extract_mfcc(y))
                            raw_waves[cname].append(y)
        return (templates, raw_waves), "Database Synchronized"

    db, msg = load_db()
    
    if db is None:
        st.error(msg)
        return

    # TABS
    t1, t2 = st.tabs(["Analyze File", "Real-time Capture"])

    audio_data = None
    file_name = ""

    with t1:
        file = st.file_uploader("Upload signal data", type=['wav', 'mp3', 'm4a'])
        if file: 
            audio_data = file.read()
            file_name = file.name

    with t2:
        rec = st.audio_input("Capture speech")
        if rec: 
            audio_data = rec.read()
            file_name = "recording.wav"

    if audio_data:
        feat_test, y_test = extractor.process(audio_data, file_name)
        
        if feat_test is not None:
            # DTW ENGINE
            with st.spinner("Executing Time Warping Analysis..."):
                distances = []
                for cname, tmpls in db[0].items():
                    for idx, t in enumerate(tmpls):
                        # Simple DTW with Window
                        d = dtw_dist_lite(feat_test, t, dtw_w)
                        distances.append((d, cname, idx))
                
                distances.sort(key=lambda x: x[0])
                
                # Class Min Distances for Anti-Bias
                class_scores = {}
                for cname in db[0].keys():
                    ds = [x[0] for x in distances if x[1] == cname]
                    class_scores[cname] = min(ds) if ds else 1e9

            # RESULTS
            best_match_name = distances[0][1]
            best_match_idx = distances[0][2]
            y_ref = db[1][best_match_name][best_match_idx]

            # METRICS DISPLAY
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Predicted Identity</div>
                        <div class="metric-value">{best_match_name.upper()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Pattern Match Score</div>
                        <div class="metric-value">{(1/(1+distances[0][0])*100):.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Active K-Factor</div>
                        <div class="metric-value">{k_val}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # WAVEFORM COMPARISON
            st.markdown("### Temporal Signal Alignment")
            fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                                subplot_titles=("Input Signal (Test)", f"Nearest Reference Pattern ({best_match_name})"))
            
            fig.add_trace(go.Scatter(y=y_test, line=dict(color='#58a6ff', width=1), name="Input"), row=1, col=1)
            fig.add_trace(go.Scatter(y=y_ref, line=dict(color='#30363d', width=1), name="Reference"), row=2, col=1)
            
            fig.update_layout(height=550, template="plotly_dark", showlegend=False,
                              margin=dict(l=20, r=20, t=50, b=20),
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(title_text="Time Samples", color="#8b949e")
            st.plotly_chart(fig, use_container_width=True)

            # RADAR & RANKING
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Similarity Distribution")
                labels = list(class_scores.keys())
                # Normalize distances to similarity percentage
                values = [1/(1+class_scores[l])*100 for l in labels]
                
                radar = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], 
                                                     fill='toself', line=dict(color='#58a6ff')))
                radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(radar, use_container_width=True)
            
            with col_b:
                st.markdown("### Comparative Ranking")
                ranked_items = sorted(class_scores.items(), key=lambda x: x[1])
                for name, dist in ranked_items:
                    sim = 1/(1+dist)*100
                    st.write(f"**{name.upper()}**")
                    st.progress(sim/100)
                    st.caption(f"Index Score: {sim:.2f}%")

def dtw_dist_lite(s1, s2, w):
    n, m = len(s1), len(s2)
    w = max(w, abs(n - m))
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    cost_mat = cdist(s1, s2, metric='euclidean')
    
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = cost_mat[i-1, j-1]
            last_min = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            dtw_matrix[i, j] = cost + last_min
            
    return dtw_matrix[n, m] / (n + m)

if __name__ == "__main__":
    main()
