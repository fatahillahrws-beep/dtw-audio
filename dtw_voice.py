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
import librosa.display
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Try to import audio recorder
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Dialect Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .class-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 3rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1e1e2e;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
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
    MAX_DURATION   = 10.0
    MIN_DURATION   = 0.5
    K_NEIGHBORS    = 5
    DTW_WINDOW     = None
    FIXED_DURATION = 5.0

# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class FeatureExtractor:
    def __init__(self, config: AudioConfig = None):
        self.cfg = config or AudioConfig()
        self.scaler = StandardScaler()
        self._fitted = False

    def load_audio(self, filepath: str):
        try:
            y, _ = librosa.load(filepath, sr=self.cfg.SAMPLE_RATE, mono=True)
            
            if len(y) / self.cfg.SAMPLE_RATE < self.cfg.MIN_DURATION:
                return None
            
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            y, _ = librosa.effects.trim(y, top_db=20)
            
            if self.cfg.FIXED_DURATION:
                target_length = int(self.cfg.FIXED_DURATION * self.cfg.SAMPLE_RATE)
                if len(y) > target_length:
                    y = y[:target_length]
                elif len(y) < target_length:
                    y = np.pad(y, (0, target_length - len(y)))
            
            return y
        except Exception as e:
            return None

    def load_from_bytes(self, audio_bytes):
        try:
            import io
            y, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.cfg.SAMPLE_RATE, mono=True)
            
            if len(y) / self.cfg.SAMPLE_RATE < self.cfg.MIN_DURATION:
                return None
            
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            y, _ = librosa.effects.trim(y, top_db=20)
            
            if self.cfg.FIXED_DURATION:
                target_length = int(self.cfg.FIXED_DURATION * self.cfg.SAMPLE_RATE)
                if len(y) > target_length:
                    y = y[:target_length]
                elif len(y) < target_length:
                    y = np.pad(y, (0, target_length - len(y)))
            
            return y
        except Exception as e:
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

    def extract_from_file(self, filepath: str):
        y = self.load_audio(filepath)
        if y is None:
            return None
        return self.extract_mfcc(y)

    def extract_from_bytes(self, audio_bytes):
        y = self.load_from_bytes(audio_bytes)
        if y is None:
            return None
        return self.extract_mfcc(y)

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
# DTW DISTANCE
# ============================================================

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window: int = None) -> float:
    n, m = len(seq1), len(seq2)
    
    if n == 0 or m == 0:
        return float('inf')
    
    cost = cdist(seq1, seq2, metric='euclidean')
    
    if np.any(np.isinf(cost)) or np.any(np.isnan(cost)):
        return float('inf')
    
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
    
    if np.isnan(result) or np.isinf(result):
        return float('inf')
    
    return result

# ============================================================
# CLASSIFIER
# ============================================================

class DTWClassifier:
    def __init__(self, config: AudioConfig = None):
        self.cfg = config or AudioConfig()
        self.extractor = FeatureExtractor(self.cfg)
        self.templates = defaultdict(list)
        self.class_names = []
        self._trained = False
        self.class_counts = {}

    def load_from_folder(self, folder_path: str):
        """Load training data langsung dari folder"""
        supported = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.opus'}
        temp_data = defaultdict(list)
        
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder_path} tidak ditemukan!")
        
        # Cari subfolder (setiap subfolder adalah nama kelas)
        for class_dir in folder.iterdir():
            if class_dir.is_dir():
                cname = class_dir.name
                audio_files = [f for f in class_dir.rglob('*') if f.suffix.lower() in supported]
                
                for af in audio_files:
                    feat = self.extractor.extract_from_file(str(af))
                    if feat is not None:
                        temp_data[cname].append(feat)
                        self.class_counts[cname] = self.class_counts.get(cname, 0) + 1
        
        # Simpan ke templates
        for cname, feats in temp_data.items():
            self.templates[cname].extend(feats)
        
        self.class_names = sorted(self.templates.keys())
        return self.class_counts

    def fit_scaler(self):
        """Fit scaler setelah semua data ditambahkan"""
        all_features = []
        for feats in self.templates.values():
            all_features.extend(feats)
        
        if all_features:
            self.extractor.fit_scaler(all_features)
            # Normalisasi semua templates
            for cname in self.templates:
                self.templates[cname] = [self.extractor.normalize(f) for f in self.templates[cname]]
            self._trained = True

    def predict(self, test_path: str = None, test_bytes: bytes = None):
        if not self._trained:
            raise RuntimeError("Belum training! Silakan train model terlebih dahulu.")
        
        if test_path:
            feat = self.extractor.extract_from_file(test_path)
        elif test_bytes:
            feat = self.extractor.extract_from_bytes(test_bytes)
        else:
            raise ValueError("Harap berikan test_path atau test_bytes")
            
        if feat is None:
            raise ValueError(f"Gagal proses audio")
        
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
        k = min(self.cfg.K_NEIGHBORS, len(all_dist))
        top_k = all_dist[:k]
        votes = Counter([x[1] for x in top_k])
        pred = votes.most_common(1)[0][0]
        
        class_avg = {}
        class_min = {}
        for cn in self.class_names:
            ds = [d for d, c, _ in all_dist if c == cn]
            class_avg[cn] = float(np.mean(ds)) if ds else float('inf')
            class_min[cn] = float(min(ds)) if ds else float('inf')
        
        # Hitung confidence
        avg_arr = np.array([class_avg[c] for c in self.class_names])
        avg_arr = np.where(np.isinf(avg_arr), 1e9, avg_arr)
        inv = 1.0 / (avg_arr + 1e-9)
        conf = inv / inv.sum()
        conf = np.nan_to_num(conf, nan=0.0)
        
        if conf.sum() == 0:
            conf = np.ones(len(self.class_names)) / len(self.class_names)
        
        class_conf = {self.class_names[i]: float(conf[i]) for i in range(len(self.class_names))}
        ranked = sorted(class_conf.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'predicted_class': pred,
            'confidence': class_conf[pred],
            'ranked_predictions': ranked,
            'class_avg_distances': class_avg,
            'class_min_distances': class_min,
            'all_distances': all_dist,
            'k_used': k,
        }

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_waveform_plot(audio_bytes):
    """Buat waveform plot interaktif dengan Plotly"""
    import io
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    time = np.arange(0, len(y)) / sr
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=y,
        mode='lines',
        name='Waveform',
        line=dict(color='#667eea', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title=dict(text='Waveform Audio', font=dict(color='white')),
        xaxis=dict(title='Waktu (detik)', gridcolor='#333', color='white'),
        yaxis=dict(title='Amplitudo', gridcolor='#333', color='white'),
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_spectrogram_plot(audio_bytes):
    """Buat spectrogram plot interaktif dengan Plotly"""
    import io
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        colorscale='Viridis',
        colorbar=dict(title='dB')
    ))
    
    fig.update_layout(
        title=dict(text='Spektrogram', font=dict(color='white')),
        xaxis=dict(title='Waktu (detik)', gridcolor='#333', color='white'),
        yaxis=dict(title='Frekuensi (Hz)', gridcolor='#333', color='white'),
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    fig.update_coloraxes(colorbar_tickfont=dict(color='white'))
    
    return fig

def create_confidence_chart(ranked_predictions):
    """Buat chart confidence interaktif"""
    classes = [r[0] for r in ranked_predictions]
    scores = [r[1] * 100 for r in ranked_predictions]
    colors = ['#2ecc71' if i == 0 else '#667eea' for i in range(len(classes))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores, y=classes,
            orientation='h',
            marker=dict(color=colors, line=dict(color='white', width=0.5)),
            text=[f'{s:.1f}%' for s in scores],
            textposition='outside',
            textfont=dict(color='white')
        )
    ])
    
    fig.update_layout(
        title=dict(text='Tingkat Kemiripan per Logat', font=dict(color='white')),
        xaxis=dict(title='Confidence Score (%)', range=[0, 100], gridcolor='#333', color='white'),
        yaxis=dict(title='Logat', gridcolor='#333', color='white'),
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_distance_chart(class_avg, class_min, predicted_class):
    """Buat chart jarak DTW interaktif"""
    classes = list(class_avg.keys())
    avg_dist = [class_avg[c] for c in classes]
    min_dist = [class_min[c] for c in classes]
    
    valid_dists = [d for d in avg_dist if not np.isinf(d)]
    max_val = max(valid_dists) if valid_dists else 10
    avg_dist = [max_val * 1.5 if np.isinf(d) else d for d in avg_dist]
    min_dist = [max_val * 1.5 if np.isinf(d) else d for d in min_dist]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Rata-rata Jarak',
        x=classes, y=avg_dist,
        marker_color='#3498db',
        text=[f'{d:.3f}' for d in avg_dist],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Jarak Minimum',
        x=classes, y=min_dist,
        marker_color='#e74c3c',
        text=[f'{d:.3f}' for d in min_dist],
        textposition='outside'
    ))
    
    if predicted_class in classes:
        pred_idx = classes.index(predicted_class)
        fig.add_vline(x=pred_idx, line_dash="dash", line_color="#2ecc71",
                      annotation_text="Prediksi", annotation_position="top")
    
    fig.update_layout(
        title=dict(text='Jarak DTW ke Tiap Kelas Logat', font=dict(color='white')),
        xaxis=dict(title='Logat', gridcolor='#333', color='white'),
        yaxis=dict(title='DTW Distance', gridcolor='#333', color='white'),
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        height=400,
        legend=dict(font=dict(color='white')),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_similarity_heatmap(all_distances, class_names):
    """Buat heatmap similarity interaktif"""
    similarities = []
    labels = []
    
    for cname in class_names:
        class_dists = [d for d, c, _ in all_distances if c == cname]
        for idx, d in enumerate(class_dists):
            if d > 0 and not np.isinf(d):
                sim = min(100, (1.0 / d) * 20)
            else:
                sim = 0
            similarities.append(sim)
            labels.append(f"{cname}<br>#{idx+1}")
    
    if not similarities:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data similarity", showarrow=False)
        return fig
    
    fig = go.Figure(data=go.Heatmap(
        z=[similarities],
        y=['Audio Test'],
        x=labels,
        colorscale='Greens',
        zmin=0, zmax=100,
        text=[[f'{s:.1f}%' for s in similarities]],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=dict(text='Heatmap Kemiripan dengan Semua Template', font=dict(color='white')),
        xaxis=dict(title='Template Sampel', tickangle=45, tickfont=dict(size=10, color='white')),
        yaxis=dict(title='', tickfont=dict(color='white')),
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_gauge_chart(confidence):
    """Buat gauge chart untuk confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Score", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': "#667eea"},
            'bgcolor': "#1e1e2e",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': '#e74c3c'},
                {'range': [33, 66], 'color': '#f0c040'},
                {'range': [66, 100], 'color': '#2ecc71'}
            ]
        },
        number={'font': {'color': 'white', 'size': 50}}
    ))
    
    fig.update_layout(
        paper_bgcolor='#1e1e2e',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Dialect Classifier</h1>
        <p>Klasifikasi Logat Suara dengan Dynamic Time Warping (DTW) + MFCC</p>
        <p>Training Data: Batak, Jawa, Melayu, Papua, Sunda</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state['classifier'] = DTWClassifier(AudioConfig())
        st.session_state['trained'] = False
        st.session_state['training_loaded'] = False
    
    clf = st.session_state['classifier']
    
    # ============================================================
    # SIDEBAR - TRAINING SECTION
    # ============================================================
    
    with st.sidebar:
        st.markdown("### 📁 Training Data")
        
        TRAINING_FOLDER = "training_data"
        
        st.markdown(f"**Folder:** `{TRAINING_FOLDER}/`")
        
        if not st.session_state['training_loaded']:
            if st.button("📂 Load Training Data", use_container_width=True, type="primary"):
                with st.spinner("Loading training data..."):
                    try:
                        class_counts = clf.load_from_folder(TRAINING_FOLDER)
                        st.session_state['training_loaded'] = True
                        st.success(f"✅ Loaded {len(class_counts)} classes!")
                        for cname, count in class_counts.items():
                            st.markdown(f"<span class='class-tag'>{cname}</span> {count} samples", 
                                       unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        if st.session_state['training_loaded'] and not st.session_state['trained']:
            st.markdown("---")
            if st.button("🚀 Train Model", use_container_width=True, type="primary"):
                with st.spinner("Training model..."):
                    clf.fit_scaler()
                    st.session_state['trained'] = True
                    st.success("✅ Model berhasil dilatih!")
                    st.rerun()
        
        if st.session_state['trained']:
            st.markdown("---")
            st.markdown("### ✅ Model Status")
            st.markdown(f"**Classes:** {', '.join(clf.class_names)}")
            for cname in clf.class_names:
                count = clf.class_counts.get(cname, 0)
                st.markdown(f"- {cname}: {count} samples")
        
        if st.session_state['trained']:
            st.markdown("---")
            if st.button("🔄 Reset Model", use_container_width=True):
                st.session_state['classifier'] = DTWClassifier(AudioConfig())
                st.session_state['trained'] = False
                st.session_state['training_loaded'] = False
                st.success("Model direset!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        k_neighbors = st.slider("K-Neighbors", 1, 10, 5)
        
        if st.button("🔄 Apply Settings", use_container_width=True):
            clf.cfg.K_NEIGHBORS = k_neighbors
            if st.session_state['trained']:
                with st.spinner("Retraining..."):
                    clf.fit_scaler()
            st.success("Settings applied!")
    
    # ============================================================
    # MAIN CONTENT
    # ============================================================
    
    if not st.session_state['trained']:
        st.info("👈 Klik 'Load Training Data' di sidebar, lalu 'Train Model'")
        st.info("""
        **Struktur folder:**
""")
return

tab1, tab2 = st.tabs(["🎤 Rekam Suara", "📁 Upload Audio"])

audio_bytes = None

with tab1:
st.markdown("### Rekam Suara Langsung")

if MIC_AVAILABLE:
    audio = mic_recorder(
        start_prompt="🎙️ Mulai Rekam",
        stop_prompt="⏹️ Berhenti",
        just_once=True,
        use_container_width=True,
        format="wav"
    )
    
    if audio:
        audio_bytes = audio['bytes']
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("🔍 Analisis Rekaman", use_container_width=True, type="primary"):
            with st.spinner("Menganalisis..."):
                try:
                    result = clf.predict(test_bytes=audio_bytes)
                    st.session_state['result'] = result
                    st.session_state['audio_bytes'] = audio_bytes
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.warning("Fitur rekam suara tidak tersedia. Gunakan tab Upload Audio.")

with tab2:
st.markdown("### Upload File Audio")

uploaded_file = st.file_uploader(
    "Pilih file audio",
    type=['wav', 'mp3', 'm4a', 'ogg', 'flac']
)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")
    
    if st.button("🔍 Analisis Audio", use_container_width=True, type="primary"):
        with st.spinner("Menganalisis..."):
            try:
                result = clf.predict(test_bytes=audio_bytes)
                st.session_state['result'] = result
                st.session_state['audio_bytes'] = audio_bytes
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Tampilkan hasil
if 'result' in st.session_state:
result = st.session_state['result']
audio_bytes = st.session_state.get('audio_bytes', None)

st.markdown("---")
st.markdown("## 📊 Hasil Klasifikasi")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{result['predicted_class']}</div>
        <div class="metric-label">Prediksi Logat</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{result['confidence']*100:.1f}%</div>
        <div class="metric-label">Confidence</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{result['k_used']}</div>
        <div class="metric-label">K-Neighbors</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">DTW+MFCC</div>
        <div class="metric-label">Metode</div>
    </div>
    """, unsafe_allow_html=True)

# Visualizations
st.markdown("### 🎵 Visualisasi Audio")
col1, col2 = st.columns(2)

with col1:
    waveform_fig = create_waveform_plot(audio_bytes)
    st.plotly_chart(waveform_fig, use_container_width=True)

with col2:
    spectrogram_fig = create_spectrogram_plot(audio_bytes)
    st.plotly_chart(spectrogram_fig, use_container_width=True)

st.markdown("### 📈 Hasil Prediksi")
col1, col2 = st.columns(2)

with col1:
    confidence_fig = create_confidence_chart(result['ranked_predictions'])
    st.plotly_chart(confidence_fig, use_container_width=True)

with col2:
    gauge_fig = create_gauge_chart(result['confidence'])
    st.plotly_chart(gauge_fig, use_container_width=True)

st.markdown("### 📉 Analisis Jarak DTW")
col1, col2 = st.columns(2)

with col1:
    distance_fig = create_distance_chart(
        result['class_avg_distances'],
        result['class_min_distances'],
        result['predicted_class']
    )
    st.plotly_chart(distance_fig, use_container_width=True)

with col2:
    heatmap_fig = create_similarity_heatmap(
        result['all_distances'],
        clf.class_names
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Export
st.markdown("### 💾 Export Hasil")

export_data = {
    'timestamp': datetime.now().isoformat(),
    'predicted_dialect': result['predicted_class'],
    'confidence_pct': result['confidence'] * 100,
    'k_neighbors': result['k_used'],
    'rankings': [{'dialect': cls, 'confidence_pct': sc * 100} 
                for cls, sc in result['ranked_predictions']]
}

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download JSON",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name=f"classification_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

with col2:
    if st.button("🔄 Analisis Baru", use_container_width=True):
        del st.session_state['result']
        if 'audio_bytes' in st.session_state:
            del st.session_state['audio_bytes']
        st.rerun()

# Footer
st.markdown("""
<div class="footer">
<p>Dialect Classifier | DTW + MFCC | k-NN Classification</p>
<p>© 2024</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
main()
        
