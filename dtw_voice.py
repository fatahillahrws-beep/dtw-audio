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
.footer {
    text-align: center;
    padding: 2rem;
    color: #666;
    font-size: 0.8rem;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin-top: 3rem;
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

    def load_from_path(self, filepath: str):
        """Membaca audio dengan aman langsung dari path disk"""
        try:
            y, _ = librosa.load(filepath, sr=self.cfg.SAMPLE_RATE, mono=True)
            
            if len(y) / self.cfg.SAMPLE_RATE < self.cfg.MIN_DURATION:
                return None
            
            # Normalisasi volume
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # Buang bagian hening
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

    def extract_from_bytes(self, audio_bytes: bytes):
        """Menyimpan byte ke file sementara di disk agar stabil dibaca librosa/ffmpeg"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            y = self.load_from_path(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
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

    def auto_load_and_train(self):
        """Otomatis mencari semua file .zip di folder yang sama dengan script"""
        supported = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.opus'}
        temp_data = defaultdict(list)
        
        # Mencari SEMUA file ZIP di direktori saat ini
        current_dir = Path('.')
        zip_files = list(current_dir.glob('*.zip'))
        
        if not zip_files:
            return False, "Tidak ditemukan file .zip di folder ini."
        
        for zip_file in zip_files:
            cname = zip_file.stem # Mengambil nama file tanpa .zip
            
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        zf.extractall(tmpdir)
                    
                    tmppath = Path(tmpdir)
                    audio_files = [f for f in tmppath.rglob('*') if f.suffix.lower() in supported]
                    
                    for af in audio_files:
                        # Abaikan file sistem sampah dari macOS / Windows
                        if '__MACOSX' in str(af) or af.name.startswith('.'):
                            continue
                            
                        try:
                            # Baca dengan fungsi disk-path yang aman
                            y = self.extractor.load_from_path(str(af))
                            if y is not None:
                                feat = self.extractor.extract_mfcc(y)
                                if feat is not None:
                                    temp_data[cname].append(feat)
                                    self.class_counts[cname] = self.class_counts.get(cname, 0) + 1
                        except Exception:
                            pass
            except Exception:
                pass # Abaikan file zip jika korup
        
        # Menggabungkan fitur
        for cname, feats in temp_data.items():
            self.templates[cname].extend(feats)
        
        self.class_names = sorted(self.templates.keys())
        
        # Fit Scaler
        all_features = []
        for feats in self.templates.values():
            all_features.extend(feats)
        
        if all_features:
            self.extractor.fit_scaler(all_features)
            for cname in self.templates:
                self.templates[cname] = [self.extractor.normalize(f) for f in self.templates[cname]]
            self._trained = True
            return True, f"Berhasil memuat dan melatih {len(self.class_names)} logat."
        else:
            return False, "Data fitur gagal diekstrak dari audio. Pastikan file ZIP berisi audio yang valid."

    def predict(self, test_bytes: bytes = None):
        if not self._trained:
            raise RuntimeError("Belum training!")
        
        feat = self.extractor.extract_from_bytes(test_bytes)
        if feat is None:
            raise ValueError("Gagal memproses audio. File mungkin rusak atau terlalu pendek.")
        
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
        
        # HITUNG CONFIDENCE BERDASARKAN JARAK RATA-RATA K-TETANGGA TERDEKAT
        class_avg = {}
        class_min = {}
        for cn in self.class_names:
            ds = [d for d, c, _ in all_dist if c == cn]
            # Menghitung rata-rata dari k-tetangga terdekat untuk kelas tersebut agar seimbang
            class_avg[cn] = float(np.mean(ds[:self.cfg.K_NEIGHBORS])) if ds else float('inf')
            class_min[cn] = float(min(ds)) if ds else float('inf')
        
        avg_arr = np.array([class_avg[c] for c in self.class_names])
        avg_arr = np.where(np.isinf(avg_arr), 1e9, avg_arr)
        inv = 1.0 / (avg_arr + 1e-9)
        conf = inv / inv.sum()
        conf = np.nan_to_num(conf, nan=0.0)
        
        if conf.sum() == 0:
            conf = np.ones(len(self.class_names)) / len(self.class_names)
        
        class_conf = {self.class_names[i]: float(conf[i]) for i in range(len(self.class_names))}
        
        # LOGIKA BARU: Prediksi mengikuti Confidence Tertinggi
        pred = max(class_conf, key=class_conf.get)
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
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        y, sr = librosa.load(tmp_path, sr=16000)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
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
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        y, sr = librosa.load(tmp_path, sr=16000)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
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
    return fig

def create_confidence_chart(ranked_predictions):
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
        height=max(300, len(classes) * 50),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def create_gauge_chart(confidence):
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

def create_similarity_heatmap(all_distances, class_names):
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
        fig.add_annotation(text="Tidak ada data valid", showarrow=False)
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
        title=dict(text='Heatmap Kemiripan', font=dict(color='white')),
        xaxis=dict(title='Template', tickangle=45, tickfont=dict(size=10, color='white')),
        yaxis=dict(title='', tickfont=dict(color='white')),
        plot_bgcolor='#1e1e2e',
        paper_bgcolor='#1e1e2e',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# ============================================================
# INITIALIZATION (AUTO-LOAD & CACHE)
# ============================================================

@st.cache_resource(show_spinner="Menyiapkan AI & Memuat ZIP Data Training (Tunggu sebentar)...")
def initialize_model():
    classifier = DTWClassifier(AudioConfig())
    success, message = classifier.auto_load_and_train()
    return classifier, success, message

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Dialect Classifier</h1>
        <p>Klasifikasi Logat Suara dengan DTW + MFCC</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Otomatis inisialisasi model
    clf, is_ready, msg = initialize_model()
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### ✅ Status Model")
        if is_ready:
            st.success("Model siap digunakan!")
            st.markdown("**Data Terintegrasi:**")
            for cname in clf.class_names:
                count = clf.class_counts.get(cname, 0)
                st.markdown(f"- **{cname.replace('Logat_', '')}**: {count} sampel")
        else:
            st.error(msg)
            
        st.markdown("---")
        st.markdown("### 🛠️ Pengelolaan Data")
        st.caption("Jika Anda baru saja menambahkan file ZIP baru ke dalam folder, klik tombol di bawah ini.")
        if st.button("🔄 Muat Ulang Semua ZIP (Refresh)", use_container_width=True):
            st.cache_resource.clear()
            if 'result' in st.session_state:
                del st.session_state['result']
            st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Pengaturan Parameter")
        new_k = st.slider("K-Neighbors", 1, 10, clf.cfg.K_NEIGHBORS)
        if new_k != clf.cfg.K_NEIGHBORS:
            clf.cfg.K_NEIGHBORS = new_k
            st.rerun()
    
    # MAIN CONTENT
    if not is_ready:
        st.warning("⚠️ Aplikasi belum bisa digunakan karena tidak ada file ZIP (misal: Logat_Batak.zip) di folder tempat script ini berjalan.")
        return

    # Tiga Tab (Upload, Rekam, Info)
    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎤 Rekam Suara", "ℹ️ Informasi"])

    with tab1:
        st.markdown("### 📂 Unggah File Suara")
        uploaded_file = st.file_uploader(
            "Pilih file audio yang ingin dianalisis (Format: WAV, MP3, M4A, OGG)",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac']
        )

        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format="audio/wav")
    
            if st.button("🔍 Mulai Analisis dari File", use_container_width=True, type="primary"):
                with st.spinner("Sedang memproses karakteristik suara..."):
                    try:
                        result = clf.predict(test_bytes=audio_bytes)
                        st.session_state['result'] = result
                        st.session_state['audio_bytes'] = audio_bytes
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal memproses file: {str(e)}")

    with tab2:
        st.markdown("### 🎤 Rekam Suara Langsung")
        st.info("Fitur ini membutuhkan izin penggunaan mikrofon di browser Anda. Klik ikon mic untuk mulai.")
        
        # Audio input native dari streamlit
        recorded_audio = st.audio_input("Rekam suara Anda di sini:")
        
        if recorded_audio is not None:
            audio_bytes = recorded_audio.read()
            
            if st.button("🔍 Mulai Analisis Rekaman", use_container_width=True, type="primary"):
                with st.spinner("Sedang memproses karakteristik suara..."):
                    try:
                        result = clf.predict(test_bytes=audio_bytes)
                        st.session_state['result'] = result
                        st.session_state['audio_bytes'] = audio_bytes
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal memproses rekaman: {str(e)}")

    with tab3:
        st.markdown("""
        **Pembaruan Algoritma:**
        - **Keamanan Format Audio:** Semua format audio kini ditulis ke *temporary storage* di hard disk sebelum diproses. Ini meminimalisir *error* untuk file `.m4a` atau `.mp3` yang diupload dari handphone.
        - **Kesesuaian Prediksi:** Keputusan algoritma sekarang 100% dipandu oleh nilai kedekatan fitur suara (*confidence score*) tertinggi, bukan sekadar jumlah *voting*. Bar hijau pada grafik akan selalu menjadi tebakan utama model.
        """)

    # HASIL PREDIKSI
    if 'result' in st.session_state:
        result = st.session_state['result']
        audio_bytes = st.session_state.get('audio_bytes')

        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['predicted_class'].replace('_', ' ')}</div>
                <div class="metric-label">Prediksi Logat Terkuat</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['confidence']*100:.1f}%</div>
                <div class="metric-label">Tingkat Keyakinan (Confidence)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['k_used']}</div>
                <div class="metric-label">K-Neighbors Terkalkulasi</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Karakteristik Audio")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_waveform_plot(audio_bytes), use_container_width=True)
        with col2:
            st.plotly_chart(create_spectrogram_plot(audio_bytes), use_container_width=True)

        st.markdown("### 🎯 Analisis Kedekatan")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_confidence_chart(result['ranked_predictions']), use_container_width=True)
        with col2:
            st.plotly_chart(create_gauge_chart(result['confidence']), use_container_width=True)

        st.markdown("### 🔥 Heatmap Kemiripan Terhadap Semua Template")
        st.plotly_chart(create_similarity_heatmap(result['all_distances'], clf.class_names), use_container_width=True)

        # Download Export
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'predicted_dialect': result['predicted_class'],
            'confidence_pct': result['confidence'] * 100,
            'rankings': [{'dialect': cls, 'confidence_pct': sc * 100} 
                        for cls, sc in result['ranked_predictions']]
        }

        st.download_button(
            label="📥 Download Laporan (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"hasil_klasifikasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        if st.button("🔄 Tutup Analisis"):
            del st.session_state['result']
            st.rerun()

    st.markdown('<div class="footer"><p>Dialect Classifier | DTW + MFCC | Rumah Data</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
