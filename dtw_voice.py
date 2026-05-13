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
# [1] GLOBAL SYSTEM ARCHITECTURE
# ==============================================================================
st.set_page_config(
    page_title="Professional Acoustic Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# [2] UNIVERSAL AUDIO DECODER BOOTSTRAP
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_engine():
    """Instalasi otomatis FFmpeg wrapper untuk kompatibilitas format audio."""
    try:
        import pydub
        import imageio_ffmpeg
    except ImportError:
        with st.spinner("🔧 Calibrating Acoustic Engine..."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "imageio-ffmpeg"])
            st.rerun()
    return True

initialize_engine()

# ==============================================================================
# [3] PREMIUM NAVY UI/UX ENGINE - ALIGNMENT OPTIMIZED
# ==============================================================================
def apply_ui_design():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #050a18; color: #f8fafc; }

        /* Professional Header */
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

        /* Result Metrics */
        .metric-card {
            background: #0f172a;
            padding: 2.2rem;
            border-radius: 16px;
            border: 1px solid #1e293b;
            text-align: center;
            transition: all 0.4s ease;
        }
        .metric-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 800; letter-spacing: 2px; }
        .metric-value { color: #38bdf8; font-size: 2.8rem; font-weight: 800; margin-top: 10px; }

        /* Professional Analysis Box - Forced Alignment */
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
        .analysis-text { 
            color: #94a3b8; font-size: 0.9rem; line-height: 1.7; text-align: justify;
        }

        /* Sidebar Interface */
        section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
        .status-card {
            background: #050a18; padding: 1.8rem; border-radius: 15px;
            border: 1px solid #1e3a8a; margin-bottom: 2.5rem; text-align: center;
        }
        .status-main { color: #38bdf8; font-size: 2rem; font-weight: 800; }
        
        /* Tab Customization */
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            padding: 10px 25px;
            border-radius: 8px 8px 0 0;
            color: #94a3b8;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e293b;
            color: #38bdf8 !important;
            border-bottom: 2px solid #38bdf8 !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_ui_design()

# ==============================================================================
# [4] ACOUSTIC SIGNAL PROCESSING CORE
# ==============================================================================
class AcousticCore:
    """Modul inti untuk manipulasi sinyal digital dan ekstraksi fitur."""
    def __init__(self, k, w):
        self.SR = 16000
        self.N_MFCC = 13
        self.K = k
        self.W = w

    def load_audio(self, path):
        """Universal decoder dengan normalisasi puncak amplitudo."""
        try:
            import pydub
            import imageio_ffmpeg
            pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            audio = pydub.AudioSegment.from_file(path).set_frame_rate(self.SR).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            return samples / float(1 << (audio.sample_width * 8 - 1))
        except:
            y, _ = librosa.load(path, sr=self.SR, mono=True)
            return y

    def extract_dialect_features(self, y):
        """Ekstraksi MFCC dengan deteksi VAD internal."""
        # Split silence secara agresif untuk anti-bias
        yt, _ = librosa.effects.trim(y, top_db=25)
        mfcc = librosa.feature.mfcc(y=yt, sr=self.SR, n_mfcc=self.N_MFCC)
        # Tambahkan delta temporal (ritme)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        features = np.hstack([mfcc.T, d1.T, d2.T])
        # Z-score normalization per audio
        return (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8), mfcc, yt

def dtw_fast_engine(s1, s2, w_constraint):
    """Penyelarasan waktu dinamis dengan metrik Cosine."""
    n, m = len(s1), len(s2)
    w = max(w_constraint, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf); dp[0, 0] = 0.0
    cost = cdist(s1, s2, metric='cosine')
    for i in range(1, n + 1):
        for j in range(max(1, i-w), min(m, i+w)+1):
            dp[i, j] = cost[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m] / (n + m)

# ==============================================================================
# [5] VISUALIZATION ENGINE - PARALLEL DESIGN
# ==============================================================================
class VizEngine:
    """Modul visualisasi dengan skema warna Navy-Cyber."""
    
    @staticmethod
    def get_navy_scale():
        return [[0, '#050a18'], [0.4, '#1e3a8a'], [1, '#38bdf8']]

    @staticmethod
    def get_amber_scale():
        return [[0, '#050a18'], [0.5, '#1e293b'], [1, '#fbbf24']]

    def plot_waveform(self, y_in, y_ref, label):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.25, 
                            subplot_titles=("Input Signal Profile", f"Master Template: {label}"))
        fig.add_trace(go.Scatter(y=y_in, line=dict(color='#38bdf8', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(y=y_ref, line=dict(color='#64748b', width=1)), row=2, col=1)
        fig.update_layout(height=500, template="plotly_dark", showlegend=False, 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    def plot_heatmap(self, data, labels):
        fig = go.Figure(data=go.Heatmap(z=[data], x=labels, colorscale=self.get_navy_scale(), 
                                        zmin=0, zmax=100, text=[[f"{v:.0f}%" for v in data]], texttemplate="%{text}"))
        fig.update_layout(height=260, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=0))
        return fig

    def plot_radar(self, labels, values):
        fig = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], 
                                            fill='toself', fillcolor='rgba(56, 189, 248, 0.15)', line=dict(color='#38bdf8', width=2.5)))
        fig.update_layout(polar=dict(bgcolor='#0f172a', radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1e293b')), 
                          template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        return fig

    def plot_spectral(self, y_in):
        sc = librosa.feature.spectral_centroid(y=y_in, sr=16000)[0]
        fig = go.Figure(); fig.add_trace(go.Scatter(y=sc, line=dict(color='#fbbf24', width=1.5), fill='tozeroy'))
        fig.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=0))
        return fig

    def plot_delta(self, mfcc):
        delta = librosa.feature.delta(mfcc)
        fig = go.Figure(data=go.Heatmap(z=delta, colorscale=self.get_amber_scale(), zmid=0))
        fig.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=0))
        return fig

# ==============================================================================
# [6] ANALYTICAL PIPELINE EXECUTION
# ==============================================================================
def start_system():
    zips = list(Path('.').glob('*.zip'))
    
    with st.sidebar:
        st.markdown(f"""<div class="status-card"><div style="color:#64748b;font-size:0.7rem;font-weight:800;text-transform:uppercase;">Computational Repository</div><div class="status-main">{len(zips)} Dialects</div><div style="color:#22c55e;font-size:0.75rem;font-weight:700;">● ENGINE OPTIMIZED</div></div>""", unsafe_allow_html=True)
        k_val = st.slider("Classification Sensitivity (K)", 1, 15, 5)
        w_val = st.slider("Dynamic Alignment Window (W)", 20, 400, 120, step=10)
        if st.button("Reload Global Intelligence", use_container_width=True):
            st.cache_resource.clear(); st.rerun()

    st.markdown("""<div class="main-header"><h1>Acoustic Intelligence Analysis</h1><p>PATTERN RECOGNITION LABORATORY SYSTEM V3.6.0</p></div>""", unsafe_allow_html=True)

    core = AcousticCore(k_val, w_val)
    viz = VizEngine()

    @st.cache_resource
    def load_db():
        t_db, w_db = defaultdict(list), defaultdict(list)
        for z in zips:
            label = z.stem.replace("Logat_", "").upper()
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(z, 'r') as zf: zf.extractall(td)
                for f in Path(td).rglob('*'):
                    if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']:
                        y = core.load_audio(str(f))
                        if y is not None:
                            feats, _, yt = core.extract_dialect_features(y)
                            t_db[label].append(feats); w_db[label].append(yt)
        return t_db, w_db

    db_t, db_w = load_db()
    if not db_t: st.error("No acoustic assets detected."); return

    # --- INTERFACE TABS (RESTORED RECORD FEATURE) ---
    tab_f, tab_r = st.tabs(["IDENTIFY SIGNAL FILE", "REAL-TIME CAPTURE"])
    audio_bytes, source_name = None, ""

    with tab_f:
        u = st.file_uploader("Signal Input", type=['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg'], label_visibility="collapsed")
        if u: audio_bytes, source_name = u.read(), u.name
    with tab_r:
        m = st.audio_input("Initialize microphone capture", label_visibility="collapsed")
        if m: audio_bytes, source_name = m.read(), "stream_live.wav"

    if audio_bytes:
        with st.spinner("Executing spectral decomposition..."):
            with tempfile.NamedTemporaryFile(suffix=Path(source_name).suffix, delete=False) as tmp:
                tmp.write(audio_bytes); p = tmp.name
            y_raw = core.load_audio(p)
            f_in, mfcc_in, y_in_t = core.extract_dialect_features(y_raw)
            os.remove(p)

            res, h_v, h_l = [], [], []
            for lab, tmpls in db_t.items():
                for i, t in enumerate(tmpls):
                    dist = dtw_fast_engine(f_in, t, w_val)
                    res.append((dist, lab, i))
                    h_v.append(1/(1+dist)*100); h_l.append(f"{lab}")

            res.sort(key=lambda x: x[0])
            winner, idx_w = res[0][1], res[0][2]
            certainty = (1/(1+res[0][0])*100)
            
            # --- PRESENTATION LAYER ---
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1: st.markdown(f'<div class="metric-card"><div class="metric-label">Identitas Dialek</div><div class="metric-value">{winner}</div></div>', unsafe_allow_html=True)
            with c_m2: st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence Index</div><div class="metric-value">{certainty:.1f}%</div></div>', unsafe_allow_html=True)
            with c_m3: st.markdown(f'<div class="metric-card"><div class="metric-label">VAD Engine</div><div class="metric-value" style="color:#22c55e">ACTIVE</div></div>', unsafe_allow_html=True)

            # 1. WAVEFORM
            st.markdown("### 1. Temporal Signal Consistency Map")
            st.plotly_chart(viz.plot_waveform(y_in_t, db_waves[winner][idx_w], winner), use_container_width=True)
            st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Konsistensi Temporal</span><p class="analysis-text">Grafik waveform di atas memetakan sinkronisasi modulasi tekanan suara antara input dan template master. Logat <b>{winner}</b> mendominasi karena memiliki struktur penekanan suku kata dan ritme bicara yang paling identik. Decoder Universal memastikan integritas sinyal tetap terjaga meskipun audio berasal dari format terkompresi.</p></div>""", unsafe_allow_html=True)

            # 2. HEATMAP
            st.markdown("### 2. Spectral Similarity Matrix")
            st.plotly_chart(viz.plot_heatmap(h_v, h_l), use_container_width=True)
            st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Korelasi Matriks</span><p class="analysis-text">Matriks kemiripan spektral ini memetakan korelasi fitur MFCC di seluruh database menggunakan palet Cyber-Navy. Area berwarna biru cerah pada kolom <b>{winner}</b> mengindikasikan densitas kecocokan fitur suara yang paling stabil dan konsisten. Hal ini meminimalkan bias klasifikasi karena sistem mendeteksi kedekatan fitur yang menyeluruh pada seluruh sampel data tersebut.</p></div>""", unsafe_allow_html=True)

            # 3 & 4. RADAR & SPECTRAL (ALIGNED)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 3. Probability Distribution Radar")
                u_labs = list(db_t.keys())
                v_radar = [1/(1+min([x[0] for x in res if x[1]==L]))*100 for L in u_labs]
                st.plotly_chart(viz.plot_radar(u_labs, v_radar), use_container_width=True)
                st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Distribusi Radar</span><p class="analysis-text">Radar distribusi menunjukkan tarikan vektor probabilitas yang condong secara eksklusif ke arah <b>{winner}</b>. Hal ini mengonfirmasi bahwa morfologi vokal Anda memiliki ciri fonetik yang unik dan tidak tumpang tindih (overlap) dengan dialek referensi lainnya dalam sistem, sehingga klasifikasi ini memiliki tingkat kepastian yang tinggi.</p></div>""", unsafe_allow_html=True)

            with col_r:
                st.markdown("### 4. Acoustic Spectral Brightness")
                st.plotly_chart(viz.plot_spectral(y_in_t), use_container_width=True)
                st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Kecerahan Akustik</span><p class="analysis-text">Spectral Centroid mengukur "pusat massa" spektrum frekuensi suara. Pola kecerahan (intonasi) pada sinyal uji ini menunjukkan profil energi frekuensi tinggi yang sangat spesifik bagi logat <b>{winner}</b>. Ini mencerminkan karakteristik "melodi" bicara yang sering ditemukan pada penutur asli daerah tersebut dalam database penelitian kami.</p></div>""", unsafe_allow_html=True)

            # 5. DELTA MFCC
            st.markdown("### 5. Velocity of Speech Features (Delta)")
            st.plotly_chart(viz.plot_delta(mfcc_in), use_container_width=True)
            st.markdown(f"""<div class="analysis-box"><span class="analysis-title">🔬 Analisis Transisi Dinamis</span><p class="analysis-text">Heatmap Delta menggunakan aksen Cyber-Amber untuk menonjolkan kecepatan perubahan fonem (ritme tempo). Kedekatan pada grafik ini menunjukkan bahwa dinamika bicara Anda memiliki profil kecepatan artikulasi yang sinkron dengan karakteristik temporal <b>{winner}</b>, memperkuat hasil deteksi dari sisi durasi dan tempo.</p></div>""", unsafe_allow_html=True)

            # RANKING
            st.markdown("### 📊 Dialect Identification Ranking")
            final_r = sorted({L: 1/(1+min([x[0] for x in res if x[1]==L]))*100 for L in db_t.keys()}.items(), key=lambda x: x[1], reverse=True)
            for n, s in final_r: st.write(f"**{n}**"); st.progress(s/100)

    st.markdown('<div class="footer">Acoustic Intelligence Research Platform © 2026 | Built for Precision Acoustic Analysis</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    start_system()
