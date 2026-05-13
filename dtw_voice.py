import streamlit as st
import os
import zipfile
import tempfile
import json
import time
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from streamlit_mic_recorder import streamlit_mic_recorder

# ==========================================
# 1. KONFIGURASI HALAMAN & UI
# ==========================================
st.set_page_config(
    page_title="AI Dialect Analyzer - Rumah Data",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan profesional
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #238636; color: white; }
    .sidebar .sidebar-content { background-image: linear-gradient(#161b22,#0d1117); }
    h1, h2, h3 { color: #58a6ff !important; }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #38444d;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOGIKA BACKEND (DTW ENGINE)
# ==========================================

class AudioConfig:
    SAMPLE_RATE = 16000
    N_MFCC = 13
    N_MELS = 40
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    N_FFT = 512
    K_NEIGHBORS = 5

class FeatureExtractor:
    def __init__(self):
        self.cfg = AudioConfig()
        self.scaler = StandardScaler()
        self._fitted = False

    def process_audio(self, y):
        # Normalisasi & Pre-processing
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # Ekstraksi MFCC + Delta + Delta2
        mfcc = librosa.feature.mfcc(y=y, sr=self.cfg.SAMPLE_RATE, n_mfcc=self.cfg.N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = np.hstack([mfcc.T, delta.T, delta2.T])
        return features

def dtw_distance(seq1, seq2):
    cost = cdist(seq1, seq2, metric='euclidean')
    n, m = len(seq1), len(seq2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i-1, j-1] + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return float(dp[n, m] / (n + m))

@st.cache_resource
class DialectModel:
    def __init__(self):
        self.cfg = AudioConfig()
        self.extractor = FeatureExtractor()
        self.templates = defaultdict(list)
        self.is_trained = False

    def train_from_zip(self, zip_path):
        all_raw = []
        temp_data = defaultdict(list)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmpdir)
            
            p = Path(tmpdir)
            # Menangani struktur zip (folder dalam folder)
            audio_paths = list(p.rglob('*.wav')) + list(p.rglob('*.mp3'))
            
            for ap in audio_paths:
                cname = ap.parent.name
                y, _ = librosa.load(str(ap), sr=self.cfg.SAMPLE_RATE)
                feat = self.extractor.process_audio(y)
                temp_data[cname].append(feat)
                all_raw.append(feat)

        # Fit Scaler
        stacked = np.vstack(all_raw)
        self.extractor.scaler.fit(stacked)
        
        for cname, feats in temp_data.items():
            for f in feats:
                norm_f = self.extractor.scaler.transform(f)
                self.templates[cname].append(norm_f)
        
        self.is_trained = True
        return list(temp_data.keys())

    def predict(self, y):
        feat = self.extractor.process_audio(y)
        feat = self.extractor.scaler.transform(feat)
        
        all_dist = []
        for cname, tmpls in self.templates.items():
            for t in tmpls:
                d = dtw_distance(feat, t)
                all_dist.append((d, cname))
        
        all_dist.sort(key=lambda x: x[0])
        top_k = all_dist[:self.cfg.K_NEIGHBORS]
        votes = Counter([x[1] for x in top_k])
        pred = votes.most_common(1)[0][0]
        
        # Hitung Confidence
        dists = np.array([x[0] for x in top_k])
        conf = (1 / (dists + 1e-6))
        conf_pct = (np.max(conf) / np.sum(conf))
        
        return pred, conf_pct, all_dist

# ==========================================
# 3. DASHBOARD UI
# ==========================================

def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3003/3003244.png", width=100)
    st.sidebar.title("Kontrol Panel")
    st.sidebar.markdown("---")
    
    # Inisialisasi Model
    model = DialectModel()
    
    # Load Training Data Otomatis
    TRAIN_ZIP = "data_training.zip"
    if os.path.exists(TRAIN_ZIP):
        with st.spinner("Mengintegrasikan Data Training..."):
            classes = model.train_from_zip(TRAIN_ZIP)
        st.sidebar.success(f"Model Siap! {len(classes)} Logat Terdeteksi.")
    else:
        st.sidebar.error("File 'data_training.zip' tidak ditemukan!")
        st.stop()

    # Header Utama
    st.title("🎙️ AI Dialect & Voice Analysis")
    st.markdown("Dashboard profesional untuk klasifikasi logat daerah berbasis **Dynamic Time Warping (DTW)**.")
    
    # Layout Kolom Input
    col1, col2 = st.columns([1, 1])
    
    audio_to_analyze = None
    
    with col1:
        st.subheader("📤 Input Suara")
        tab1, tab2 = st.tabs(["Upload File", "Rekam Langsung"])
        
        with tab1:
            uploaded_file = st.file_uploader("Pilih file audio (WAV/MP3)", type=["wav", "mp3"])
            if uploaded_file:
                audio_to_analyze, _ = librosa.load(uploaded_file, sr=16000)
                st.audio(uploaded_file)
        
        with tab2:
            st.write("Klik ikon mic untuk mulai merekam:")
            audio_rec = streamlit_mic_recorder(
                start_prompt="Mulai Rekam 🎤",
                stop_prompt="Berhenti & Simpan ⏹️",
                key='recorder'
            )
            if audio_rec:
                audio_to_analyze = librosa.util.buf_to_float(audio_rec['bytes'], n_bytes=2)
                st.audio(audio_rec['bytes'])

    # Logika Analisis
    if audio_to_analyze is not None:
        with col2:
            st.subheader("📊 Hasil Analisis")
            with st.spinner("Menganalisis karakteristik suara..."):
                start_time = time.time()
                pred, conf, all_dist = model.predict(audio_to_analyze)
                duration = time.time() - start_time
            
            # Tampilan Card Prediksi
            st.markdown(f"""
                <div class="prediction-card">
                    <p style="color: #8b949e; margin-bottom: 0;">Logat Terdeteksi</p>
                    <h1 style="margin: 0; font-size: 3em;">{pred.upper()}</h1>
                    <p style="color: #3fb950; font-weight: bold; font-size: 1.2em;">
                        Confidence: {conf*100:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Waktu komputasi: {duration:.2f} detik")

        # --- VISUALISASI ---
        st.markdown("---")
        st.subheader("📈 Visualisasi Kedalaman Data")
        
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            # Waveform & Spectrogram
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            fig.patch.set_facecolor('#0e1117')
            
            # Waveform
            librosa.display.waveshow(audio_to_analyze, sr=16000, ax=ax[0], color='#58a6ff')
            ax[0].set_title("Audio Waveform", color='white')
            ax[0].set_facecolor('#161b22')
            
            # Mel-Spectrogram
            S = librosa.feature.melspectrogram(y=audio_to_analyze, sr=16000, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=16000, ax=ax[1])
            ax[1].set_title("Mel-Spectrogram", color='white')
            plt.colorbar(img, ax=ax[1], format='%+2.0f dB')
            
            plt.tight_layout()
            st.pyplot(fig)

        with v_col2:
            # Bar Chart Jarak
            st.write("**Perbandingan Jarak DTW per Kelas**")
            # Ambil rata-rata jarak per kelas untuk grafik
            dist_summary = defaultdict(list)
            for d, c in all_dist:
                dist_summary[c].append(d)
            
            avg_dists = {k: np.mean(v) for k, v in dist_summary.items()}
            sorted_avg = dict(sorted(avg_dists.items(), key=lambda item: item[1]))
            
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            fig_bar.patch.set_facecolor('#0e1117')
            ax_bar.set_facecolor('#161b22')
            
            colors = sns.color_palette("viridis", len(sorted_avg))
            sns.barplot(x=list(sorted_avg.values()), y=list(sorted_avg.keys()), palette=colors, ax=ax_bar)
            
            ax_bar.set_xlabel("Jarak DTW (Semakin Kecil Semakin Mirip)", color='white')
            ax_bar.tick_params(colors='white')
            st.pyplot(fig_bar)

        # Tabel Ranking
        with st.expander("Lihat Detail Ranking Logat"):
            st.table([{"Logat": k, "Rata-rata Jarak": f"{v:.4f}"} for k, v in sorted_avg.items()])

    else:
        st.info("Silakan upload file atau gunakan fitur rekam suara untuk memulai analisis.")

if __name__ == "__main__":
    main()
