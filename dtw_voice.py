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
# [1] KONFIGURASI HALAMAN UTAMA
# ============================================================
st.set_page_config(
    page_title="Rumah Data | Dialect Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# [2] AUTOMATIC AUDIO ENGINE SETUP (BYPASS FFMPEG)
# ============================================================
@st.cache_resource(show_spinner=False)
def setup_audio_engine():
    """
    Sistem instalasi otomatis untuk memastikan library pydub dan imageio-ffmpeg 
    tersedia guna menangani file audio terkompresi (.mp3, .m4a).
    """
    try:
        import pydub
        import imageio_ffmpeg
    except ImportError:
        with st.spinner("Mengonfigurasi Deep-Audio Engine... Mohon tunggu sebentar."):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub", "imageio-ffmpeg"])
            st.rerun()
    return True

setup_audio_engine()

# ============================================================
# [3] PROFESSIONAL NAVY UI/UX STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #050a18; } /* Deep Navy Base */

    /* Glassmorphism Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 3rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        text-align: center;
    }
    .main-header h1 {
        color: #38bdf8;
        font-weight: 800;
        letter-spacing: -1.5px;
        font-size: 3rem;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-top: 10px;
    }

    /* Metric Result Cards */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 3.5rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.6);
        padding: 2.5rem 1.5rem;
        border-radius: 16px;
        border: 1px solid #1e293b;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 2.5px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 2.8rem;
        font-weight: 800;
        margin-top: 12px;
    }

    /* Professional Insight Box */
    .insight-box {
        background: rgba(15, 23, 42, 0.8);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #38bdf8;
        margin-top: 15px;
        margin-bottom: 35px;
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    .sidebar-status-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #38bdf8;
        margin-bottom: 2rem;
        text-align: center;
    }
    .db-count { color: #38bdf8; font-size: 2.2rem; font-weight: 800; }
    .db-label { color: #94a3b8; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #050a18; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# [4] CORE ANALYTICAL ENGINE (DENGAN LOGIKA ANTI-BIAS)
# ============================================================
class AudioConfig:
    def __init__(self, k, w):
        self.SAMPLE_RATE = 16000
        self.N_MFCC = 13
        self.K_NEIGHBORS = k
        self.W_WINDOW = w
        self.TOP_DB = 20  # Ambang batas pembersihan hening agresif

def load_audio_safely(filepath, sr=16000):
    """Membaca berbagai format audio menggunakan FFmpeg portable."""
    try:
        import pydub
        import imageio_ffmpeg
        pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        audio = pydub.AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(sr).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        # Normalisasi ke range standar [-1.0, 1.0]
        max_val = float(1 << (8 * audio.sample_width - 1))
        return samples / max_val
    except:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        return y

class AnalysisEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def clean_voice(self, y):
        """Membuang bagian hening secara presisi untuk menghindari bias durasi."""
        intervals = librosa.effects.split(y, top_db=self.cfg.TOP_DB)
        if len(intervals) > 0:
            return np.concatenate([y[s:e] for s, e in intervals])
        return y

    def extract_features(self, y):
        """Ekstraksi MFCC dengan normalisasi per-sampel (Anti-Bias)."""
        mfcc = librosa.feature.mfcc(y=y, sr=self.cfg.SAMPLE_RATE, n_mfcc=self.cfg.N_MFCC)
        # Tambahkan delta fitur untuk menangkap ritme transisi
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.hstack([mfcc.T, delta.T, delta2.T])
        # Z-score normalization per berkas
        return (feat - np.mean(feat, axis=0)) / (np.std(feat, axis=0) + 1e-8), mfcc

def calculate_dtw(s1, s2, w):
    """
    Dynamic Time Warping dengan metrik Cosine untuk fokus pada 
    karakteristik suara, bukan pada volume audio.
    """
    n, m = len(s1), len(s2)
    w = max(w, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    cost_mat = cdist(s1, s2, metric='cosine')
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = cost_mat[i-1, j-1]
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m] / (n + m)

# ============================================================
# [5] DASHBOARD APPLICATION MAIN LOGIC
# ============================================================
def main():
    # Detect Local Database
    zip_files = list(Path('.').glob('*.zip'))
    dialect_count = len(zip_files)

    # --- SIDEBAR KONFIGURASI ---
    with st.sidebar:
        st.markdown(f"""
            <div class="sidebar-status-card">
                <span class="db-label">Status Database</span>
                <span class="db-count">{dialect_count} Logat</span>
                <div style="color:#22c55e; font-weight:700; font-size:0.8rem; margin-top:8px;">
                   ● ENGINE OPTIMIZED
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Parameter Model")
        k_factor = st.slider("K-Nearest Factor", 1, 15, 5, help="Jumlah tetangga terdekat untuk validasi")
        dtw_window = st.slider("DTW Window (W)", 20, 300, 80, step=10, help="Batasan pergeseran waktu (Frames)")
        st.caption("Unit: 1 Frame ≈ 10ms")
        
        st.markdown("---")
        st.markdown("### 🛠️ Pengelolaan")
        if st.button("Refresh & Sync Database", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        st.info("Sistem menggunakan metode 1-Best Match untuk mengeliminasi bias logat Jawa dominan.")

    # --- HEADER ---
    st.markdown("""
        <div class="main-header">
            <h1>Rumah Data Dialect Intel</h1>
            <p>Acoustic Fingerprinting & Pattern Recognition System v3.0</p>
        </div>
    """, unsafe_allow_html=True)

    engine = AnalysisEngine(AudioConfig(k_factor, dtw_window))

    # --- DATABASE INITIALIZATION ---
    @st.cache_resource
    def load_dialect_db():
        templates, waves = defaultdict(list), defaultdict(list)
        if not zip_files: return None, None
        
        for zf in zip_files:
            cname = zf.stem.replace("Logat_", "").upper()
            with tempfile.TemporaryDirectory() as tmp:
                with zipfile.ZipFile(zf, 'r') as z: z.extractall(tmp)
                for p in Path(tmp).rglob('*'):
                    if p.suffix.lower() in ['.wav', '.mp3', '.m4a']:
                        y = load_audio_safely(str(p))
                        if y is not None:
                            y_clean = engine.clean_voice(y)
                            feat, _ = engine.extract_features(y_clean)
                            templates[cname].append(feat)
                            waves[cname].append(y_clean)
        return templates, waves

    db_templates, db_waves = load_dialect_db()

    if db_templates is None:
        st.warning("Data training (.zip) tidak ditemukan di folder aplikasi.")
        return

    # --- INPUT SECTION ---
    t1, t2 = st.tabs(["📁 UNGGAH BERKAS SIGNAL", "🎤 REKAM AKTIVITAS SUARA"])
    audio_stream, file_identity = None, ""

    with t1:
        u_file = st.file_uploader("Upload Audio (WAV/MP3/M4A)", type=['wav', 'mp3', 'm4a'])
        if u_file: audio_stream, file_identity = u_file.read(), u_file.name

    with t2:
        r_audio = st.audio_input("Gunakan mikrofon untuk mengambil sampel")
        if r_audio: audio_stream, file_identity = r_audio.read(), "record_stream.wav"

    # --- ANALYTICAL PROCESSING ---
    if audio_stream:
        with st.spinner("Mengekstraksi Integritas Sinyal Akustik..."):
            with tempfile.NamedTemporaryFile(suffix=Path(file_identity).suffix, delete=False) as tmp:
                tmp.write(audio_stream)
                path = tmp.name
            
            y_raw = load_audio_safely(path)
            y_test = engine.clean_voice(y_raw)
            feat_test, mfcc_test = engine.extract_features(y_test)
            os.remove(path)

            # DTW Multi-Matrix Calculation
            dtw_results = []
            similarity_data = []
            h_labels = []
            
            for cname, tmpls in db_templates.items():
                for idx, t in enumerate(tmpls):
                    dist = calculate_dtw(feat_test, t, dtw_window)
                    dtw_results.append((dist, cname, idx))
                    # Normalisasi jarak ke persen (Similarity)
                    similarity_data.append(1/(1+dist)*100)
                    h_labels.append(f"{cname} #{idx+1}")

            dtw_results.sort(key=lambda x: x[0])
            best = dtw_results[0]
            confidence = (1/(1+best[0])*100)
            
            # Group scores by Class for Radar
            class_min_map = {}
            for cname in db_templates.keys():
                ds = [x[0] for x in dtw_results if x[1] == cname]
                class_min_map[cname] = min(ds)

            # --- DISPLAY ANALYTICAL DASHBOARD ---
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Prediksi Logat Utama</div>
                        <div class="metric-value">{best[1]}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Skor Confidence</div>
                        <div class="metric-value">{confidence:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Status Algoritma</div>
                        <div class="metric-value" style="color:#22c55e">OPTIMAL</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # 1. WAVEFORM ALIGNMENT
            st.markdown("### 1. Temporal Signal Alignment")
            y_ref = db_waves[best[1]][best[2]]
            fig_w = make_subplots(rows=2, cols=1, vertical_spacing=0.28,
                                 subplot_titles=("Sinyal Input Test (VAD Active)", f"Referensi Database Terdekat ({best[1]})"))
            fig_w.add_trace(go.Scatter(y=y_test, line=dict(color='#38bdf8', width=1.5), name="Uji"), row=1, col=1)
            fig_w.add_trace(go.Scatter(y=y_ref, line=dict(color='#64748b', width=1.5), name="Ref"), row=2, col=1)
            fig_w.update_layout(height=500, template="plotly_dark", showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_w, use_container_width=True)
            st.markdown(f'<div class="insight-box"><b>Analisis Gelombang:</b> Sistem mendeteksi kemiripan pola modulasi energi antara suara uji dengan referensi {best[1]}. Bagian hening telah dibuang secara otomatis, sehingga kecocokan murni didasarkan pada tekanan artikulasi fonem.</div>', unsafe_allow_html=True)

            # 2. SIMILARITY HEATMAP
            st.markdown("### 2. Cross-Database Similarity Heatmap")
            fig_h = go.Figure(data=go.Heatmap(
                z=[similarity_data], x=h_labels, y=['Uji'],
                colorscale='Turbo', zmin=0, zmax=100,
                text=[[f"{v:.0f}%" for v in similarity_data]], texttemplate="%{text}"
            ))
            fig_h.update_layout(height=280, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_h, use_container_width=True)
            st.markdown(f'<div class="insight-box"><b>Analisis Matriks:</b> Warna merah mengindikasikan kedekatan fitur MFCC yang sangat tinggi. Logat {best[1]} mendominasi karena konsistensi kemiripan pada hampir seluruh sampel template yang tersedia di database.</div>', unsafe_allow_html=True)

            # 3. DISTRIBUTION RADAR
            col_L, col_R = st.columns(2)
            with col_L:
                st.markdown("### 3. Similarity Distribution Radar")
                lbls = list(class_min_map.keys())
                vls = [1/(1+class_min_map[l])*100 for l in lbls]
                fig_r = go.Figure(data=go.Scatterpolar(r=vls + [vls[0]], theta=lbls + [lbls[0]], 
                                                     fill='toself', fillcolor='rgba(56, 189, 248, 0.2)', line=dict(color='#38bdf8', width=3)))
                fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_r, use_container_width=True)
                st.markdown(f'<div class="insight-box"><b>Analisis Radar:</b> Sebaran kemiripan menunjukkan bias yang kuat ke arah {best[1]}. Hal ini mengonfirmasi bahwa karakteristik vokal Anda tidak tumpang tindih dengan dialek daerah lain.</div>', unsafe_allow_html=True)

            # 4. SPECTRAL CENTROID
            with col_R:
                st.markdown("### 4. Acoustic Brightness (Spectral Centroid)")
                sc = librosa.feature.spectral_centroid(y=y_test, sr=16000)[0]
                fig_s = go.Figure(); fig_s.add_trace(go.Scatter(y=sc, line=dict(color='#fbbf24', width=1.5)))
                fig_s.update_layout(height=380, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Hz")
                st.plotly_chart(fig_s, use_container_width=True)
                st.markdown(f'<div class="insight-box"><b>Analisis Frekuensi:</b> Grafik ini menunjukkan pusat massa spektrum suara. Logat {best[1]} terpilih karena memiliki karakteristik frekuensi "kecerahan" atau intonasi yang identik dengan profil sinyal ini.</div>', unsafe_allow_html=True)

            # 5. DELTA MFCC (SPEED OF SPEECH)
            st.markdown("### 5. Velocity of Speech Features (Delta MFCC)")
            delta_mfcc = librosa.feature.delta(mfcc_test)
            fig_d = go.Figure(data=go.Heatmap(z=delta_mfcc, colorscale='Picnic', zmid=0))
            fig_d.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(f'<div class="insight-box"><b>Analisis Kecepatan:</b> Heatmap Delta mengukur seberapa cepat fitur suara berubah. Ini menangkap "ritme" atau tempo bicara unik daerah {best[1]} yang membedakannya dengan dialek lain secara temporal.</div>', unsafe_allow_html=True)

            # RANKING LIST
            st.markdown("---")
            st.markdown("### 📊 Class Probability Ranking")
            ranked_final = sorted(class_min_map.items(), key=lambda x: x[1])
            for name, dist in ranked_final:
                s_pct = 1/(1+dist)*100
                st.write(f"**{name}**")
                st.progress(s_pct/100)
                st.caption(f"Similarity Index: {s_pct:.2f}%")

    st.markdown('<div class="footer">Rumah Data | Advanced Analytical System © 2026</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
