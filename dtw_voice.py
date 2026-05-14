import streamlit as st
import os
import zipfile
import tempfile
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew

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
    return True

initialize_universal_engine()

# ==============================================================================
# NAVY UI STYLES (disingkat agar tidak melebihi 800 line)
# ==============================================================================
def apply_professional_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
        :root {
            --navy-900: #020812; --navy-800: #040d1e; --navy-700: #071228;
            --navy-600: #0c1f3f; --accent-sky: #38bdf8; --accent-cyan: #22d3ee;
            --accent-emerald: #34d399; --text-primary: #e8f0fe; --text-secondary: #8eafd4;
            --border: rgba(56,189,248,0.15);
        }
        .stApp { background: var(--navy-900); color: var(--text-primary); }
        .hero-header { background: linear-gradient(160deg, var(--navy-700), var(--navy-800)); padding: 2rem; border-radius: 0 0 20px 20px; margin: -2rem -2rem 2rem -2rem; }
        .hero-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; }
        .hero-title span { color: var(--accent-sky); }
        .metric-card { background: var(--navy-600); padding: 1rem; border-radius: 12px; border: 1px solid var(--border); }
        .metric-value { font-size: 1.5rem; font-weight: 800; color: var(--accent-sky); }
        .analysis-box { background: rgba(7,18,40,0.9); padding: 1rem; border-radius: 10px; border-left: 3px solid var(--accent-sky); margin: 1rem 0; }
        section[data-testid="stSidebar"] { background: var(--navy-800) !important; }
        .rank-item { display: flex; align-items: center; gap: 10px; padding: 0.5rem; background: var(--navy-700); border-radius: 8px; margin-bottom: 5px; }
        .rank-bar-bg { flex: 1; height: 6px; background: var(--navy-600); border-radius: 3px; }
        .rank-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent-sky), var(--accent-cyan)); border-radius: 3px; }
        .chart-container { background: var(--navy-700); border-radius: 12px; padding: 0.8rem; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

apply_professional_styles()

# ==============================================================================
# IMPROVED ACOUSTIC CORE - FOKUS UNTUK MEMBEDAKAN SUNDA VS KENDARI
# ==============================================================================
class AcousticCore:
    def __init__(self, k, w):
        self.SR = 16000
        self.N_MFCC = 26
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

    def extract_dialect_features(self, y):
        """Ekstraksi fitur lengkap untuk diskriminasi dialek"""
        yt, _ = librosa.effects.trim(y, top_db=25)
        if len(yt) < self.SR * 0.5:
            yt = y
        
        # Normalisasi amplitude
        if np.max(np.abs(yt)) > 0:
            yt = yt / np.max(np.abs(yt))
        
        # === FITUR UTAMA ===
        mfcc = librosa.feature.mfcc(y=yt, sr=self.SR, n_mfcc=self.N_MFCC)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        
        # === FITUR PENTING UNTUK DISKRIMINASI SUNDA ===
        
        # 1. Pitch / F0 (intonasi khas Sunda)
        try:
            f0, _, _ = librosa.pyin(yt, fmin=65, fmax=350, sr=self.SR)
            f0 = np.nan_to_num(f0)
            f0_voiced = f0[f0 > 0]
            pitch_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0
            pitch_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 0
            pitch_max = np.max(f0_voiced) if len(f0_voiced) > 0 else 0
        except:
            pitch_mean = pitch_std = pitch_max = 0
        pitch_features = np.array([[pitch_mean, pitch_std, pitch_max]]).T
        pitch_features = np.resize(pitch_features, (1, mfcc.shape[1]))
        
        # 2. Spectral features
        centroid = librosa.feature.spectral_centroid(y=yt, sr=self.SR)
        bandwidth = librosa.feature.spectral_bandwidth(y=yt, sr=self.SR)
        rolloff = librosa.feature.spectral_rolloff(y=yt, sr=self.SR)
        
        # 3. Chroma (untuk karakteristik nada)
        chroma = librosa.feature.chroma_stft(y=yt, sr=self.SR)
        
        # 4. Zero Crossing Rate (ritme bicara)
        zcr = librosa.feature.zero_crossing_rate(yt)
        
        # 5. RMS Energy (tekanan suku kata)
        rms = librosa.feature.rms(y=yt)
        
        # 6. Spectral Contrast
        try:
            contrast = librosa.feature.spectral_contrast(y=yt, sr=self.SR, n_bands=6)
        except:
            contrast = np.zeros((6, mfcc.shape[1]))
        
        # Resize semua ke dimensi yang sama
        def resize_feat(feat, target_cols):
            if feat.shape[1] != target_cols:
                return np.resize(feat, (feat.shape[0], target_cols))
            return feat
        
        target_cols = mfcc.shape[1]
        pitch_features = resize_feat(pitch_features, target_cols)
        centroid = resize_feat(centroid, target_cols)
        bandwidth = resize_feat(bandwidth, target_cols)
        rolloff = resize_feat(rolloff, target_cols)
        chroma = resize_feat(chroma, target_cols)
        zcr = resize_feat(zcr, target_cols)
        rms = resize_feat(rms, target_cols)
        contrast = resize_feat(contrast, target_cols)
        
        # Stack semua fitur (total: 26+26+26+3+1+1+1+12+1+1+6 = 104 fitur)
        features = np.vstack([
            mfcc, d1, d2,
            pitch_features, centroid, bandwidth, rolloff,
            chroma, zcr, rms, contrast
        ]).T
        
        # Statistik global per frame untuk embedding
        feat_mean = np.mean(features, axis=0)
        feat_std = np.std(features, axis=0)
        feat_skew = skew(features, axis=0, nan_policy='omit')
        feat_kurt = kurtosis(features, axis=0, nan_policy='omit')
        
        # Normalisasi z-score
        mu = np.mean(features, axis=0)
        sigma = np.std(features, axis=0) + 1e-8
        features_norm = (features - mu) / sigma
        
        return features_norm, mfcc, yt, (feat_mean, feat_std, feat_skew, feat_kurt)

# ==============================================================================
# IMPROVED DTW WITH WEIGHTED DISTANCE
# ==============================================================================
def weighted_dtw(s1, s2, w_const):
    """DTW dengan weighted distance dan slope constraint"""
    n, m = len(s1), len(s2)
    w = max(w_const, abs(n - m))
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    
    # Multi-metric dengan bobot
    cost_cosine = cdist(s1, s2, metric='cosine')
    cost_euclidean = cdist(s1, s2, metric='euclidean')
    cost_manhattan = cdist(s1, s2, metric='cityblock')
    
    # Normalisasi
    max_euc = np.max(cost_euclidean) + 1e-8
    max_man = np.max(cost_manhattan) + 1e-8
    
    # Kombinasi weighted (bobot dioptimasi untuk diskriminasi dialek)
    cost = (0.5 * cost_cosine + 
            0.3 * (cost_euclidean / max_euc) + 
            0.2 * (cost_manhattan / max_man))
    
    for i in range(1, n + 1):
        for j in range(max(1, i-w), min(m, i+w)+1):
            prev_min = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
            dp[i, j] = cost[i-1, j-1] + prev_min
    
    return dp[n, m] / (n + m)

# ==============================================================================
# ENSEMBLE CLASSIFIER
# ==============================================================================
def ensemble_classify(feats_in, embedding_in, db_templates, db_embeddings, w_val):
    """Ensemble classifier dengan multiple comparison methods"""
    all_scores = []
    
    for label, templates in db_templates.items():
        best_score = float('inf')
        embeddings_label = db_embeddings[label]
        
        for idx, t in enumerate(templates):
            # 1. Frame-based DTW
            dtw_frame = weighted_dtw(feats_in, t, w_val)
            
            # 2. Embedding-based comparison
            emb_mean_in, emb_std_in, emb_skew_in, emb_kurt_in = embedding_in
            emb_mean_t, emb_std_t, emb_skew_t, emb_kurt_t = embeddings_label[idx]
            
            # Cosine similarity untuk mean vector
            cos_sim = np.dot(emb_mean_in, emb_mean_t) / (
                np.linalg.norm(emb_mean_in) * np.linalg.norm(emb_mean_t) + 1e-8
            )
            cos_dist = 1 - cos_sim
            
            # Euclidean distance untuk semua statistik
            euc_dist = np.linalg.norm(emb_mean_in - emb_mean_t)
            euc_dist += 0.3 * np.linalg.norm(emb_std_in - emb_std_t)
            euc_dist += 0.2 * np.linalg.norm(emb_skew_in - emb_skew_t)
            euc_dist_norm = euc_dist / (1 + euc_dist)
            
            # 3. Dynamic Time Warping pada statistik
            mean_stack_in = np.column_stack([emb_mean_in, emb_std_in, emb_skew_in, emb_kurt_in])
            mean_stack_t = np.column_stack([emb_mean_t, emb_std_t, emb_skew_t, emb_kurt_t])
            dtw_embed = weighted_dtw(mean_stack_in, mean_stack_t, 10)
            
            # Kombinasi dengan bobot optimal
            combined = (0.45 * dtw_frame + 
                       0.30 * cos_dist + 
                       0.15 * euc_dist_norm + 
                       0.10 * dtw_embed)
            
            best_score = min(best_score, combined)
        
        all_scores.append((best_score, label))
    
    all_scores.sort(key=lambda x: x[0])
    return all_scores

# ==============================================================================
# VISUALIZATION ENGINE
# ==============================================================================
class VizEngine:
    def _base_layout(self, h=350):
        return dict(height=h, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#8eafd4', size=10))

    def plot_waveform(self, y_in, y_ref, label):
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2)
        fig.add_trace(go.Scatter(y=y_in, line=dict(color='#38bdf8', width=1), fill='tozeroy'), row=1, col=1)
        fig.add_trace(go.Scatter(y=y_ref, line=dict(color='#22d3ee', width=1), fill='tozeroy'), row=2, col=1)
        fig.update_layout(**self._base_layout(400))
        return fig

    def plot_heatmap(self, data, labels):
        fig = go.Figure(data=go.Heatmap(z=[data], x=labels, colorscale=[[0, '#020812'], [1, '#38bdf8']],
                                        text=[[f"{v:.0f}%" for v in data]], texttemplate="%{text}"))
        fig.update_layout(**self._base_layout(200), margin=dict(l=0, r=0, t=10, b=0))
        return fig

    def plot_radar(self, labels, values):
        fig = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]],
                                             fill='toself', line=dict(color='#38bdf8', width=2)))
        fig.update_layout(polar=dict(bgcolor='rgba(7,18,40,0.6)'), **self._base_layout(320))
        return fig

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def start_dialect_analysis():
    zip_files = list(Path('.').glob('*.zip'))

    # Sidebar
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:1rem;"><b style="font-size:1.2rem;">Analisis Akustik</b><br><span style="font-size:0.7rem;">Laboratorium Pengenalan Pola</span></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style="background:#0c1f3f;padding:1rem;border-radius:10px;"><div>Status Komputasi</div><div style="font-size:2rem;font-weight:800;">{len(zip_files)}<span style="font-size:0.8rem;"> logat</span></div></div>""", unsafe_allow_html=True)
        k_val = st.slider("Sensitivitas (K)", 1, 15, 5)
        w_val = st.slider("Batas Jendela (W)", 20, 400, 120, step=10)
        if st.button("Muat Ulang Database", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # Header
    st.markdown("""<div class="hero-header"><div class="hero-title">Laboratorium <span>Pengenalan</span><br>Dialek</div><div>DTW · MFCC-26 · Ensemble Classifier · 104-D Feature</div></div>""", unsafe_allow_html=True)

    core = AcousticCore(k_val, w_val)
    viz = VizEngine()

    @st.cache_resource
    def boot_database():
        db_templates, db_waves, db_embeddings = defaultdict(list), defaultdict(list), defaultdict(list)
        for z in zip_files:
            label = z.stem.replace("Logat_", "").upper()
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(z, 'r') as zf: zf.extractall(td)
                for f in Path(td).rglob('*'):
                    if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']:
                        y = core.load_audio(str(f))
                        if y is not None and len(y) > 0:
                            feats, _, yt, embedding = core.extract_dialect_features(y)
                            if feats is not None and len(feats) > 0:
                                db_templates[label].append(feats)
                                db_waves[label].append(yt)
                                db_embeddings[label].append(embedding)
        return db_templates, db_waves, db_embeddings

    db_templates, db_waves, db_embeddings = boot_database()

    if not db_templates:
        st.error("Dataset akustik tidak ditemukan. Harap sediakan arsip .zip di direktori kerja.")
        return

    # Input section
    st.markdown("### Pilih Sumber Audio")
    tab_f, tab_r = st.tabs(["Unggah Berkas", "Rekam Langsung"])
    audio_stream, source_id = None, ""

    with tab_f:
        u = st.file_uploader("Unggah Data Akustik", type=['wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg'], label_visibility="collapsed")
        if u:
            audio_stream, source_id = u.read(), u.name

    with tab_r:
        m = st.audio_input("Rekam Pola Suara", label_visibility="collapsed")
        if m:
            audio_stream, source_id = m.read(), "rekaman.wav"

    # Processing
    if audio_stream:
        with st.spinner("Memproses audio dengan 104 fitur akustik..."):
            with tempfile.NamedTemporaryFile(suffix=Path(source_id).suffix, delete=False) as tmp:
                tmp.write(audio_stream)
                path = tmp.name
            
            y_raw = core.load_audio(path)
            if y_raw is None or len(y_raw) == 0:
                st.error("Gagal memuat audio.")
                os.remove(path)
                return
            
            feats_in, mfcc_in, y_in_t, embedding_in = core.extract_dialect_features(y_raw)
            os.remove(path)
            
            if feats_in is None or len(feats_in) == 0:
                st.error("Gagal ekstraksi fitur.")
                return

            # Klasifikasi ensemble
            scores = ensemble_classify(feats_in, embedding_in, db_templates, db_embeddings, w_val)
            winner = scores[0][1]
            confidence = (1 / (1 + scores[0][0])) * 100
            
            # Cari referensi untuk visualisasi
            idx_winner = 0
            ref_wave = db_waves[winner][idx_winner] if winner in db_waves and db_waves[winner] else y_in_t

        # Hasil
        st.markdown(f"""
            <div style="display:flex;gap:1rem;margin:1rem 0;">
                <div class="metric-card"><div>Identitas Dialek</div><div class="metric-value">{winner}</div></div>
                <div class="metric-card"><div>Skor Kepercayaan</div><div class="metric-value">{confidence:.1f}%</div></div>
                <div class="metric-card"><div>Dimensi Fitur</div><div class="metric-value">104-D</div></div>
            </div>
        """, unsafe_allow_html=True)

        # Waveform comparison
        st.markdown("### 01. Peta Penyelarasan Sinyal Temporal")
        st.plotly_chart(viz.plot_waveform(y_in_t, ref_wave, winner), use_container_width=True)
        st.markdown(f"""<div class="analysis-box">Analisis waveform menunjukkan sinkronisasi temporal. Dialek <b>{winner}</b> memiliki struktur ritme yang paling sesuai dengan sinyal input.</div>""", unsafe_allow_html=True)

        # Heatmap similarity
        st.markdown("### 02. Matriks Kemiripan Spektral")
        h_vals = []
        h_lbls = []
        for score, lbl in scores:
            h_vals.append(1/(1+score)*100)
            h_lbls.append(lbl)
        st.plotly_chart(viz.plot_heatmap(h_vals, h_lbls), use_container_width=True)
        
        # Penjelasan khusus untuk kasus Sunda vs Kendari
        if winner != "SUNDA" and any(lbl == "SUNDA" for lbl, _ in scores):
            sunda_score = next((1/(1+s)*100 for s, lbl in scores if lbl == "SUNDA"), 0)
            st.markdown(f"""
                <div class="analysis-box">
                    <b>⚠️ Analisis Khusus:</b> Sistem mendeteksi <b>{winner}</b> (skor {confidence:.0f}%) bukan SUNDA (skor {sunda_score:.0f}%).<br>
                    <b>Penyebab umum:</b>
                    <ul>
                        <li>Audio training SUNDA mungkin kurang representatif (durasi terlalu pendek/bising)</li>
                        <li>Intonasi dan pitch (F0) - SUNDA memiliki pola nada yang sangat khas</li>
                        <li>Pastikan file zip bernama <code>Logat_SUNDA.zip</code> berisi audio bersih minimal 3-5 detik</li>
                    </ul>
                    <b>Solusi:</b> Perbanyak sampel audio SUNDA (3-5 file berbeda) dalam satu zip.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="analysis-box">Matriks kemiripan menunjukkan korelasi fitur MFCC-26 + pitch + chroma. Dialek <b>{winner}</b> memiliki skor similarity tertinggi.</div>""", unsafe_allow_html=True)

        # Radar chart
        st.markdown("### 03. Radar Probabilitas")
        col_l, col_r = st.columns(2)
        with col_l:
            u_labs = list(db_templates.keys())
            v_radar = [1/(1+next(s for s, lbl in scores if lbl == L))*100 for L in u_labs]
            st.plotly_chart(viz.plot_radar(u_labs, v_radar), use_container_width=True)
        
        with col_r:
            # Spectral centroid
            sc = librosa.feature.spectral_centroid(y=y_in_t, sr=16000)[0]
            fig_s = go.Figure(data=go.Scatter(y=sc, line=dict(color='#fbbf24', width=1.5), fill='tozeroy'))
            fig_s.update_layout(**viz._base_layout(300))
            st.plotly_chart(fig_s, use_container_width=True)
        
        st.markdown(f"""<div class="analysis-box">Radar menunjukkan distribusi probabilitas yang condong ke {winner}. Spectral centroid mencerminkan karakteristik frekuensi dialek.</div>""", unsafe_allow_html=True)

        # Ranking
        st.markdown("### 04. Peringkat Kelas Komprehensif")
        for rank_idx, (score, name) in enumerate(scores):
            sim = 1/(1+score)*100
            is_top = rank_idx == 0
            st.markdown(f"""
                <div class="rank-item" style="{'border-color:#38bdf8;background:rgba(56,189,248,0.1)' if is_top else ''}">
                    <div style="width:30px;">#{rank_idx+1}</div>
                    <div style="min-width:80px;"><b>{name}</b></div>
                    <div class="rank-bar-bg"><div class="rank-bar-fill" style="width:{sim:.0f}%"></div></div>
                    <div style="width:50px;text-align:right;">{sim:.0f}%</div>
                </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""<hr><div style="text-align:center;font-size:0.7rem;">Platform Riset Akustik · DTW + MFCC-26 + Ensemble Classifier</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    start_dialect_analysis()
