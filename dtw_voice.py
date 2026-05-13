import streamlit as st
import os
import zipfile
import tempfile
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import librosa
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Dialect Classifier - Rumah Data",
    page_icon="🎙️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
    text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.main-header h1 { color: white; font-size: 2.8rem; margin-bottom: 0.5rem; font-weight: 800; }
.main-header p { color: #e0e0e0; font-size: 1.1rem; }
.metric-card {
    background: linear-gradient(135deg, #16222A 0%, #3A6073 100%);
    border-radius: 15px; padding: 1.5rem; text-align: center;
    border: 1px solid #4a90e2; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}
.metric-value { font-size: 2.2rem; font-weight: bold; color: #00f2fe; }
.metric-label { color: #cbd5e1; font-size: 1rem; margin-top: 0.5rem; font-weight: 500; }
.footer { text-align: center; padding: 2rem; color: #888; font-size: 0.9rem; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONFIGURASI AUDIO
# ============================================================
class AudioConfig:
    SAMPLE_RATE    = 16000
    N_MFCC         = 13
    N_MELS         = 40
    HOP_LENGTH     = 160
    WIN_LENGTH     = 400
    N_FFT          = 512
    DELTA          = True
    DELTA_DELTA    = True
    MIN_DURATION   = 0.1  
    K_NEIGHBORS    = 5
    DTW_WINDOW     = None
    FIXED_DURATION = None

# ============================================================
# PEMBERSIH SUARA (SUPER VAD - Pembunuh Hening)
# ============================================================
def clean_audio_voice_only(y, sr):
    """Membuang semua noise statis & durasi hening secara agresif"""
    # Normalisasi awal untuk membaca puncak suara sesungguhnya
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
        
    # Split audio: hanya ambil bagian yang bunyinya > 15 dB dari noise floor
    intervals = librosa.effects.split(y, top_db=15)
    
    if len(intervals) > 0:
        # Gabungkan potongan-potongan yang ada suaranya saja
        y_clean = np.concatenate([y[start:end] for start, end in intervals])
        # Jika setelah dibersihkan masih ada sisa minimal 0.1 detik, gunakan yang bersih
        if len(y_clean) >= int(0.1 * sr):
            return y_clean
            
    return y

# ============================================================
# FEATURE EXTRACTOR
# ============================================================
class FeatureExtractor:
    def __init__(self, config: AudioConfig = None):
        self.cfg = config or AudioConfig()
        self.scaler = StandardScaler()
        self._fitted = False

    def load_from_path(self, filepath: str):
        try:
            y, _ = librosa.load(filepath, sr=self.cfg.SAMPLE_RATE, mono=True)
            
            # Terapkan pembersih hening tingkat tinggi
            y = clean_audio_voice_only(y, self.cfg.SAMPLE_RATE)
            
            if len(y) / self.cfg.SAMPLE_RATE < self.cfg.MIN_DURATION:
                return None
            
            return y
        except Exception:
            return None

    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        mfcc = librosa.feature.mfcc(
            y=y, sr=cfg.SAMPLE_RATE, n_mfcc=cfg.N_MFCC,
            n_mels=cfg.N_MELS, hop_length=cfg.HOP_LENGTH,
            win_length=cfg.WIN_LENGTH, n_fft=cfg.N_FFT
        )
        features = [mfcc.T]
        if cfg.DELTA:
            features.append(librosa.feature.delta(mfcc, order=1).T)
        if cfg.DELTA_DELTA:
            features.append(librosa.feature.delta(mfcc, order=2).T)
            
        return np.hstack(features)

    def extract_from_bytes(self, audio_bytes: bytes, file_name: str = 'temp.wav'):
        ext = Path(file_name).suffix if Path(file_name).suffix else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            y = self.load_from_path(tmp_path)
        except Exception:
            y = None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        if y is None:
            return None, None
        return self.extract_mfcc(y), y

    def fit_scaler(self, all_features: list):
        stacked = np.vstack(all_features)
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(stacked)
        self._fitted = True

    def normalize(self, features: np.ndarray) -> np.ndarray:
        if self._fitted:
            result = self.scaler.transform(features)
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result
        return features

# ============================================================
# DTW DISTANCE (MENGGUNAKAN COSINE DISTANCE ANTI-BIAS)
# ============================================================
def dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window: int = None) -> float:
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0: return float('inf')
    
    # [PENTING] Cosine distance fokus pada "bentuk ritme", bukan volume keras/pelan
    cost = cdist(seq1, seq2, metric='cosine')
    if np.any(np.isinf(cost)) or np.any(np.isnan(cost)): return float('inf')
    
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    
    if window is None:
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i, j] = cost[i-1, j-1] + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    else:
        for i in range(1, n + 1):
            j_min = max(1, i - window)
            j_max = min(m, i + window)
            for j in range(j_min, j_max + 1):
                dp[i, j] = cost[i-1, j-1] + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    
    result = float(dp[n, m] / (n + m))
    if np.isnan(result) or np.isinf(result): return float('inf')
    return result

# ============================================================
# CLASSIFIER
# ============================================================
class DTWClassifier:
    def __init__(self, config: AudioConfig = None):
        self.cfg = config or AudioConfig()
        self.extractor = FeatureExtractor(self.cfg)
        self.templates = defaultdict(list)
        self.raw_audios = defaultdict(list) 
        self.class_names = []
        self._trained = False
        self.class_counts = {}
        self.error_log = [] 

    def auto_load_and_train(self):
        supported = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
        temp_data = defaultdict(list)
        self.error_log = [] 
        
        zip_files = [p for p in Path('.').iterdir() if p.suffix.lower() == '.zip']
        
        if not zip_files:
            return False, "Tidak ditemukan file .zip di folder ini."
        
        for zip_file in zip_files:
            cname = zip_file.stem
            
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        zf.extractall(tmpdir)
                    
                    audio_files = [f for f in Path(tmpdir).rglob('*') if f.suffix.lower() in supported]
                    
                    for af in audio_files:
                        if '__MACOSX' in str(af) or af.name.startswith('.'):
                            continue
                            
                        try:
                            y = self.extractor.load_from_path(str(af))
                            if y is not None:
                                feat = self.extractor.extract_mfcc(y)
                                if feat is not None:
                                    temp_data[cname].append(feat)
                                    self.raw_audios[cname].append(y) 
                                    self.class_counts[cname] = self.class_counts.get(cname, 0) + 1
                                else:
                                    self.error_log.append(f"Gagal ekstrak MFCC: {af.name}")
                            else:
                                self.error_log.append(f"Audio sangat pendek / rusak: {af.name}")
                        except Exception as e:
                            self.error_log.append(f"Format tidak terbaca: {af.name} | Error: {str(e)}")
            except Exception as e:
                self.error_log.append(f"Gagal mengekstrak ZIP {zip_file.name}: {str(e)}")
        
        for cname, feats in temp_data.items():
            self.templates[cname].extend(feats)
        
        self.class_names = sorted(self.templates.keys())
        all_features = [f for feats in self.templates.values() for f in feats]
        
        if all_features:
            self.extractor.fit_scaler(all_features)
            for cname in self.templates:
                self.templates[cname] = [self.extractor.normalize(f) for f in self.templates[cname]]
            self._trained = True
            return True, f"Berhasil memuat {len(self.class_names)} logat dengan total {sum(self.class_counts.values())} sampel."
        else:
            return False, "Gagal mengekstrak fitur. Pastikan format audio adalah .WAV atau Anda telah menginstal FFmpeg di sistem."

    def predict(self, test_bytes: bytes, file_name: str = 'temp.wav'):
        if not self._trained:
            raise RuntimeError("Belum training!")
        
        feat, test_y = self.extractor.extract_from_bytes(test_bytes, file_name)
        if feat is None:
            raise ValueError("Gagal memproses audio uji. Format mungkin tidak didukung atau durasinya menjadi 0 detik setelah dibersihkan.")
        
        feat = self.extractor.normalize(feat)
        
        all_dist = []
        for cname in self.class_names:
            for idx, tmpl in enumerate(self.templates[cname]):
                d = dtw_distance(feat, tmpl, window=self.cfg.DTW_WINDOW)
                if not np.isinf(d):
                    all_dist.append((d, cname, idx))
        
        if not all_dist:
            raise RuntimeError("Tidak ada distance valid!")
        
        all_dist.sort(key=lambda x: x[0])
        
        best_match = all_dist[0]
        if hasattr(self, 'raw_audios') and best_match[1] in self.raw_audios and len(self.raw_audios[best_match[1]]) > best_match[2]:
            ref_y = self.raw_audios[best_match[1]][best_match[2]]
        else:
            ref_y = np.zeros_like(test_y) 
        
        # [PENTING] LOGIKA ANTI-BIAS: 1-Best Match Scoring
        class_min = {}
        for cn in self.class_names:
            ds = [d for d, c, _ in all_dist if c == cn]
            class_min[cn] = float(min(ds)) if ds else float('inf')
        
        min_arr = np.array([class_min[c] for c in self.class_names])
        min_arr = np.where(np.isinf(min_arr), 1e9, min_arr)
        
        # Perhitungan Persentase Logis (Inversi nilai Cosine)
        sim_arr = 1.0 / (1.0 + min_arr)
        conf = sim_arr / sim_arr.sum()
        conf = np.nan_to_num(conf, nan=0.0)
        
        if conf.sum() == 0:
            conf = np.ones(len(self.class_names)) / len(self.class_names)
        
        class_conf = {self.class_names[i]: float(conf[i]) for i in range(len(self.class_names))}
        pred = max(class_conf, key=class_conf.get)
        ranked = sorted(class_conf.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'predicted_class': pred,
            'confidence': class_conf[pred],
            'ranked_predictions': ranked,
            'all_distances': all_dist,
            'k_used': 1, # Menggunakan pendekatan 1-Best-Match per Kelas
            'test_waveform': test_y,
            'ref_waveform': ref_y,
            'ref_class_name': best_match[1]
        }

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def create_waveform_comparison(test_y, ref_y, ref_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.1,
                        subplot_titles=("Sinyal Suara Anda", f"Sinyal Logat Referensi Terdekat ({ref_name.replace('Logat_','')})"))

    t_test = np.arange(len(test_y)) / 16000
    t_ref = np.arange(len(ref_y)) / 16000

    fig.add_trace(go.Scatter(x=t_test, y=test_y, mode='lines', line=dict(color='#00f2fe', width=1.5), name='Uji', fill='tozeroy', fillcolor='rgba(0, 242, 254, 0.2)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_ref, y=ref_y, mode='lines', line=dict(color='#4facfe', width=1.5), name='Ref', fill='tozeroy', fillcolor='rgba(79, 172, 254, 0.2)'), row=2, col=1)

    fig.update_layout(title=dict(text='Perbandingan Bentuk Gelombang Waktu', font=dict(color='white')), plot_bgcolor='#1e1e2e', paper_bgcolor='#1e1e2e', height=400, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    fig.update_xaxes(title_text="Waktu (detik)", color="white")
    return fig

def create_spectrogram_plot(y):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig = go.Figure(data=go.Heatmap(z=D, colorscale='Magma', colorbar=dict(title='dB')))
    fig.update_layout(title=dict(text='Spektrogram Suara Anda', font=dict(color='white')), plot_bgcolor='#1e1e2e', paper_bgcolor='#1e1e2e', height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_confidence_chart(ranked_predictions):
    classes = [r[0].replace('Logat_', '') for r in ranked_predictions]
    scores = [r[1] * 100 for r in ranked_predictions]
    colors = ['#00f2fe' if i == 0 else '#3A6073' for i in range(len(classes))]
    
    fig = go.Figure(data=[go.Bar(x=scores, y=classes, orientation='h', marker=dict(color=colors, line=dict(color='white', width=1)), text=[f'{s:.1f}%' for s in scores], textposition='outside', textfont=dict(color='white'))])
    fig.update_layout(title=dict(text='Persentase Keyakinan Klasifikasi', font=dict(color='white')), xaxis=dict(range=[0, 110], gridcolor='#333'), yaxis=dict(autorange="reversed"), plot_bgcolor='#1e1e2e', paper_bgcolor='#1e1e2e', height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_radar_chart(ranked_predictions):
    classes = [r[0].replace('Logat_', '') for r in ranked_predictions]
    scores = [r[1] * 100 for r in ranked_predictions]
    classes.append(classes[0])
    scores.append(scores[0])
    
    fig = go.Figure(data=go.Scatterpolar(r=scores, theta=classes, fill='toself', line=dict(color='#00f2fe', width=3), fillcolor='rgba(0, 242, 254, 0.3)', marker=dict(color='white', size=8)))
    fig.update_layout(title=dict(text='Distribusi Kekerabatan Logat', font=dict(color='white')), polar=dict(bgcolor='#16222A', radialaxis=dict(visible=True, range=[0, 100], gridcolor='#444', color='white'), angularaxis=dict(gridcolor='#555', color='white')), showlegend=False, plot_bgcolor='#1e1e2e', paper_bgcolor='#1e1e2e', height=380, margin=dict(l=40, r=40, t=60, b=40))
    return fig

def create_similarity_heatmap(all_distances, class_names):
    valid_dists = [d for d, _, _ in all_distances if not np.isinf(d)]
    if not valid_dists: return go.Figure()
        
    min_d, max_d = min(valid_dists), max(valid_dists)
    range_d = max_d - min_d if max_d > min_d else 1.0
    similarities, labels = [], []
    
    for cname in class_names:
        class_dists = [d for d, c, _ in all_distances if c == cname]
        for idx, d in enumerate(class_dists):
            sim = 100 * (1 - ((d - min_d) / range_d)) if not np.isinf(d) else 0
            similarities.append(sim)
            labels.append(f"{cname.replace('Logat_', '')} #{idx+1}")
    
    fig = go.Figure(data=go.Heatmap(z=[similarities], y=['Audio Uji'], x=labels, colorscale='Turbo', zmin=0, zmax=100, text=[[f'{s:.0f}%' for s in similarities]], texttemplate='%{text}', textfont={"size": 10, "color": "white"}))
    fig.update_layout(title=dict(text='Heatmap Relatif Database (Merah = Sangat Mirip)', font=dict(color='white')), xaxis=dict(tickangle=-45, tickfont=dict(size=10, color='#cbd5e1')), plot_bgcolor='#1e1e2e', paper_bgcolor='#1e1e2e', height=300, margin=dict(l=20, r=20, t=50, b=80))
    return fig

# ============================================================
# MAIN APP
# ============================================================
@st.cache_resource(show_spinner="Menyiapkan AI & Membaca ZIP Data Training...")
def initialize_model():
    classifier = DTWClassifier(AudioConfig())
    success, message = classifier.auto_load_and_train()
    return classifier, success, message

def main():
    st.markdown("""<div class="main-header"><h1>🎙️ Dialect Classifier</h1><p>Klasifikasi Logat Suara Berbasis DTW + MFCC</p></div>""", unsafe_allow_html=True)
    
    clf, is_ready, msg = initialize_model()
    
    if not hasattr(clf, 'error_log') or not hasattr(clf, 'raw_audios'):
        st.cache_resource.clear()
        st.rerun()
    
    with st.sidebar:
        st.markdown("### ✅ Status Model")
        if is_ready:
            st.success("Model siap digunakan!")
            st.markdown("**Total Data Terbaca per Logat:**")
            for cname in clf.class_names:
                count = clf.class_counts.get(cname, 0)
                st.markdown(f"- **{cname.replace('Logat_', '')}**: {count} sampel")
                
            if hasattr(clf, 'error_log') and clf.error_log:
                with st.expander("⚠️ Info File Gagal Terbaca", expanded=False):
                    for err in clf.error_log:
                        st.caption(err)
        else:
            st.error(msg)
            
        st.markdown("---")
        st.markdown("### 🛠️ Pengelolaan Data")
        if st.button("🔄 Muat Ulang Semua ZIP (Refresh)", use_container_width=True):
            st.cache_resource.clear()
            if 'result' in st.session_state: del st.session_state['result']
            st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Info Algoritma")
        st.info("🔥 **PEMBARUAN SUPER VAD AKTIF!** Seluruh durasi hening dan *noise* kosong di-ekstraksi habis. DTW sekarang beroperasi dengan mode 'Cosine Distance' anti-bias volume.")
    
    if not is_ready:
        st.warning("⚠️ Belum ada file ZIP terdeteksi. Taruh file ZIP ke dalam folder script ini lalu klik 'Muat Ulang'.")
        return

    tab1, tab2 = st.tabs(["📁 Upload & Analisis", "🎤 Rekam Suara Langsung"])

    with tab1:
        st.markdown("### 📂 Unggah File Suara")
        uploaded_file = st.file_uploader("Upload file logat misterius (Direkomendasikan .WAV)", type=['wav', 'mp3', 'm4a', 'ogg', 'flac'])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            file_name = uploaded_file.name
            st.audio(audio_bytes)
            if st.button("🔍 Mulai Analisis Audio", use_container_width=True, type="primary"):
                with st.spinner("Membuang noise hening dan Melakukan Time Warping..."):
                    try:
                        st.session_state['result'] = clf.predict(test_bytes=audio_bytes, file_name=file_name)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal memproses file: {str(e)}")

    with tab2:
        st.markdown("### 🎤 Rekam Suara")
        recorded_audio = st.audio_input("Bicara sesuatu menggunakan logat Anda:")
        if recorded_audio is not None:
            audio_bytes = recorded_audio.read()
            if st.button("🔍 Analisis Hasil Rekaman", use_container_width=True, type="primary"):
                with st.spinner("Membuang noise hening dan Melakukan Time Warping..."):
                    try:
                        st.session_state['result'] = clf.predict(test_bytes=audio_bytes, file_name='rekaman.wav')
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal memproses rekaman: {str(e)}")

    if 'result' in st.session_state:
        result = st.session_state['result']

        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis Algoritma DTW")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{result["predicted_class"].replace("Logat_", "").upper()}</div><div class="metric-label">Prediksi Logat Utama</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{result["confidence"]*100:.1f}%</div><div class="metric-label">Skor Confidence</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">✓</div><div class="metric-label">Anti-Bias & VAD Aktif</div></div>', unsafe_allow_html=True)

        st.markdown("### 📈 Visualisasi Jarak Waktu (Time Warping)")
        st.caption("Jika Anda melihat gambar di bawah ini lebih pendek durasinya, itu karena sistem berhasil membuang seluruh keheningan yang mengganggu.")
        st.plotly_chart(create_waveform_comparison(result['test_waveform'], result['ref_waveform'], result['ref_class_name']), use_container_width=True)

        st.markdown("### 🎯 Rasio Kedekatan Multi-Kelas")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_confidence_chart(result['ranked_predictions']), use_container_width=True)
        with col2:
            st.plotly_chart(create_radar_chart(result['ranked_predictions']), use_container_width=True)

        st.markdown("### 🔥 Skala Distribusi ke Seluruh Model Database")
        st.plotly_chart(create_similarity_heatmap(result['all_distances'], clf.class_names), use_container_width=True)

        st.markdown("### 🎵 Pemetaan Spektrogram Input")
        st.plotly_chart(create_spectrogram_plot(result['test_waveform']), use_container_width=True)

        if st.button("🔄 Tutup Analisis", use_container_width=True):
            del st.session_state['result']
            st.rerun()

    st.markdown('<div class="footer"><p>Rumah Data | Dialect Classifier v2.0 | Berbasis DTW + MFCC</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
