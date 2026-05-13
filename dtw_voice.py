import streamlit as st
import os
import zipfile
import tempfile
import json
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import librosa
import soundfile as sf
import audioread

from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dialect Classifier",
    page_icon="🎙️",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.main-header{
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    padding:2rem;
    border-radius:20px;
    text-align:center;
    margin-bottom:2rem;
}

.main-header h1{
    color:white;
    font-size:2.7rem;
}

.metric-card{
    background:#1e1e2e;
    border-radius:15px;
    padding:1rem;
    text-align:center;
}

.metric-value{
    font-size:2rem;
    font-weight:bold;
    color:#667eea;
}

.metric-label{
    color:#ccc;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# AUDIO CONFIG
# ============================================================

class AudioConfig:

    SAMPLE_RATE = 16000

    N_MFCC = 13

    N_MELS = 40

    HOP_LENGTH = 160

    WIN_LENGTH = 400

    N_FFT = 512

    DELTA = True

    DELTA_DELTA = True

    MIN_DURATION = 0.05

    FIXED_DURATION = 5.0

    K_NEIGHBORS = 5

    DTW_WINDOW = None

# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class FeatureExtractor:

    def __init__(self, config=None):

        self.cfg = config or AudioConfig()

        self.scaler = StandardScaler()

        self._fitted = False

    # ========================================================
    # LOAD AUDIO SUPER STABIL
    # ========================================================

    def load_from_path(self, filepath):

        try:

            filepath = str(filepath)

            if not os.path.exists(filepath):
                return None

            if os.path.getsize(filepath) < 500:
                return None

            y = None

            # ====================================================
            # LIBROSA
            # ====================================================

            try:

                y, sr = librosa.load(
                    filepath,
                    sr=self.cfg.SAMPLE_RATE,
                    mono=True,
                    res_type='kaiser_fast'
                )

            except Exception:
                y = None

            # ====================================================
            # SOUND FILE
            # ====================================================

            if y is None:

                try:

                    y, sr = sf.read(filepath)

                    if len(y.shape) > 1:
                        y = np.mean(y, axis=1)

                    if sr != self.cfg.SAMPLE_RATE:

                        y = librosa.resample(
                            y,
                            orig_sr=sr,
                            target_sr=self.cfg.SAMPLE_RATE
                        )

                except Exception:
                    y = None

            # ====================================================
            # AUDIOREAD + FFMPEG
            # ====================================================

            if y is None:

                try:

                    samples = []

                    with audioread.audio_open(filepath) as f:

                        sr = f.samplerate

                        for buf in f:

                            data = np.frombuffer(
                                buf,
                                dtype=np.int16
                            )

                            samples.append(data)

                    if samples:

                        y = np.concatenate(samples)

                        y = y.astype(np.float32) / 32768.0

                        if sr != self.cfg.SAMPLE_RATE:

                            y = librosa.resample(
                                y,
                                orig_sr=sr,
                                target_sr=self.cfg.SAMPLE_RATE
                            )

                except Exception as e:

                    print(f"GAGAL AUDIOREAD: {filepath}")

                    print(e)

                    return None

            # ====================================================
            # VALIDASI
            # ====================================================

            if y is None:
                return None

            y = np.asarray(y, dtype=np.float32)

            y = np.nan_to_num(y)

            if len(y) == 0:
                return None

            # ====================================================
            # TRIM SILENCE
            # ====================================================

            try:

                y, _ = librosa.effects.trim(
                    y,
                    top_db=45
                )

            except Exception:
                pass

            duration = len(y) / self.cfg.SAMPLE_RATE

            if duration < self.cfg.MIN_DURATION:
                return None

            # ====================================================
            # NORMALISASI
            # ====================================================

            max_amp = np.max(np.abs(y))

            if max_amp > 0:
                y = y / max_amp

            # ====================================================
            # FIXED DURATION
            # ====================================================

            if self.cfg.FIXED_DURATION:

                target_length = int(
                    self.cfg.FIXED_DURATION *
                    self.cfg.SAMPLE_RATE
                )

                if len(y) > target_length:

                    y = y[:target_length]

                else:

                    y = np.pad(
                        y,
                        (0, target_length - len(y)),
                        mode='constant'
                    )

            return y

        except Exception as e:

            print(f"ERROR LOAD AUDIO: {filepath}")

            print(e)

            return None

    # ========================================================
    # MFCC
    # ========================================================

    def extract_mfcc(self, y):

        try:

            cfg = self.cfg

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=cfg.SAMPLE_RATE,
                n_mfcc=cfg.N_MFCC,
                n_mels=cfg.N_MELS,
                hop_length=cfg.HOP_LENGTH,
                win_length=cfg.WIN_LENGTH,
                n_fft=cfg.N_FFT
            )

            features = [mfcc.T]

            if cfg.DELTA:

                features.append(
                    librosa.feature.delta(
                        mfcc,
                        order=1
                    ).T
                )

            if cfg.DELTA_DELTA:

                features.append(
                    librosa.feature.delta(
                        mfcc,
                        order=2
                    ).T
                )

            result = np.hstack(features)

            result = np.nan_to_num(result)

            return result

        except Exception:

            return None

    # ========================================================
    # EXTRACT FROM BYTES
    # ========================================================

    def extract_from_bytes(self, audio_bytes):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.wav'
        ) as tmp:

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

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def fit_scaler(self, all_features):

        stacked = np.vstack(all_features)

        stacked = np.nan_to_num(stacked)

        self.scaler.fit(stacked)

        self._fitted = True

    def normalize(self, features):

        if not self._fitted:
            return features

        result = self.scaler.transform(features)

        return np.nan_to_num(result)

# ============================================================
# DTW
# ============================================================

def dtw_distance(seq1, seq2, window=None):

    n, m = len(seq1), len(seq2)

    if n == 0 or m == 0:
        return float('inf')

    cost = cdist(seq1, seq2, metric='euclidean')

    if np.any(np.isnan(cost)):
        return float('inf')

    dp = np.full((n + 1, m + 1), np.inf)

    dp[0, 0] = 0

    for i in range(1, n + 1):

        if window is None:

            j_start = 1
            j_end = m + 1

        else:

            j_start = max(1, i - window)
            j_end = min(m + 1, i + window)

        for j in range(j_start, j_end):

            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1]
            )

    result = dp[n, m] / (n + m)

    if np.isnan(result):
        return float('inf')

    return float(result)

# ============================================================
# CLASSIFIER
# ============================================================

class DTWClassifier:

    def __init__(self, config=None):

        self.cfg = config or AudioConfig()

        self.extractor = FeatureExtractor(self.cfg)

        self.templates = defaultdict(list)

        self.class_names = []

        self.class_counts = {}

        self.failed_files = []

        self._trained = False

    # ========================================================
    # VALIDATE ZIP
    # ========================================================

    def is_zip_valid(self, zip_path):

        try:

            with zipfile.ZipFile(zip_path, 'r') as zf:

                bad = zf.testzip()

            return bad is None

        except Exception:

            return False

    # ========================================================
    # AUTO LOAD ZIP
    # ========================================================

    def auto_load_and_train(self):

        supported = {
            '.wav',
            '.mp3',
            '.m4a',
            '.ogg',
            '.flac',
            '.opus'
        }

        temp_data = defaultdict(list)

        current_dir = Path('.')

        zip_files = []

        for ext in ['*.zip', '*.ZIP']:

            zip_files.extend(
                current_dir.rglob(ext)
            )

        if not zip_files:

            return False, "ZIP tidak ditemukan"

        total_audio = 0

        for zip_file in zip_files:

            cname = zip_file.stem.strip()

            cname = cname.replace(' ', '_')

            cname = cname.replace('-', '_')

            print("=" * 50)

            print(f"MEMBACA ZIP: {zip_file.name}")

            if not self.is_zip_valid(zip_file):

                print("ZIP RUSAK")

                continue

            try:

                with tempfile.TemporaryDirectory() as tmpdir:

                    with zipfile.ZipFile(zip_file, 'r') as zf:

                        zf.extractall(tmpdir)

                    tmppath = Path(tmpdir)

                    audio_files = []

                    for f in tmppath.rglob('*'):

                        try:

                            if not f.is_file():
                                continue

                            if '__MACOSX' in str(f):
                                continue

                            if f.name.startswith('.'):
                                continue

                            if f.suffix.lower() not in supported:
                                continue

                            audio_files.append(f)

                        except Exception:
                            continue

                    print(f"TOTAL AUDIO: {len(audio_files)}")

                    success_count = 0

                    for af in audio_files:

                        try:

                            y = self.extractor.load_from_path(
                                str(af)
                            )

                            if y is None:

                                self.failed_files.append(
                                    str(af)
                                )

                                continue

                            feat = self.extractor.extract_mfcc(y)

                            if feat is None:

                                self.failed_files.append(
                                    str(af)
                                )

                                continue

                            if np.isnan(feat).any():

                                self.failed_files.append(
                                    str(af)
                                )

                                continue

                            temp_data[cname].append(feat)

                            success_count += 1

                            total_audio += 1

                        except Exception as e:

                            self.failed_files.append(
                                f"{af} -> {e}"
                            )

                    self.class_counts[cname] = success_count

                    print(f"BERHASIL: {success_count}")

            except Exception:

                traceback.print_exc()

        # ====================================================
        # TRAIN
        # ====================================================

        for cname, feats in temp_data.items():

            self.templates[cname].extend(feats)

        self.class_names = sorted(
            self.templates.keys()
        )

        all_features = []

        for feats in self.templates.values():

            all_features.extend(feats)

        if not all_features:

            return False, "Tidak ada fitur valid"

        self.extractor.fit_scaler(all_features)

        for cname in self.templates:

            self.templates[cname] = [

                self.extractor.normalize(f)

                for f in self.templates[cname]
            ]

        self._trained = True

        return (
            True,
            f"Berhasil membaca {total_audio} audio dari {len(self.class_names)} logat"
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, audio_bytes):

        if not self._trained:

            raise RuntimeError("Model belum training")

        feat = self.extractor.extract_from_bytes(
            audio_bytes
        )

        if feat is None:

            raise ValueError(
                "Audio gagal diproses"
            )

        feat = self.extractor.normalize(feat)

        all_dist = []

        for cname in self.class_names:

            for idx, tmpl in enumerate(
                self.templates[cname]
            ):

                d = dtw_distance(
                    feat,
                    tmpl,
                    self.cfg.DTW_WINDOW
                )

                if not np.isinf(d):

                    all_dist.append(
                        (d, cname, idx)
                    )

        if not all_dist:

            raise RuntimeError(
                "Tidak ada distance valid"
            )

        all_dist.sort(key=lambda x: x[0])

        class_avg = {}

        for cn in self.class_names:

            ds = [
                d for d, c, _
                in all_dist
                if c == cn
            ]

            if ds:

                class_avg[cn] = float(
                    np.mean(
                        ds[:self.cfg.K_NEIGHBORS]
                    )
                )

            else:

                class_avg[cn] = float('inf')

        avg_arr = np.array(
            [
                class_avg[c]
                for c in self.class_names
            ]
        )

        avg_arr = np.where(
            np.isinf(avg_arr),
            1e9,
            avg_arr
        )

        inv = 1.0 / (avg_arr + 1e-9)

        conf = inv / inv.sum()

        class_conf = {

            self.class_names[i]: float(conf[i])

            for i in range(len(self.class_names))
        }

        pred = max(
            class_conf,
            key=class_conf.get
        )

        ranked = sorted(
            class_conf.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {

            'predicted_class': pred,

            'confidence': class_conf[pred],

            'ranked_predictions': ranked
        }

# ============================================================
# VISUALIZATION
# ============================================================

def create_confidence_chart(ranked_predictions):

    classes = [
        r[0]
        for r in ranked_predictions
    ]

    scores = [
        r[1] * 100
        for r in ranked_predictions
    ]

    fig = go.Figure()

    fig.add_trace(

        go.Bar(
            x=scores,
            y=classes,
            orientation='h',
            text=[
                f"{s:.1f}%"
                for s in scores
            ],
            textposition='outside'
        )
    )

    fig.update_layout(
        height=400
    )

    return fig

# ============================================================
# CACHE
# ============================================================

@st.cache_resource(
    ttl=0,
    show_spinner="Membaca semua ZIP..."
)
def initialize_model():

    classifier = DTWClassifier(
        AudioConfig()
    )

    success, message = (
        classifier.auto_load_and_train()
    )

    return classifier, success, message

# ============================================================
# MAIN APP
# ============================================================

def main():

    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Dialect Classifier</h1>
        <p>DTW + MFCC + Multi Audio Loader</p>
    </div>
    """, unsafe_allow_html=True)

    clf, is_ready, msg = initialize_model()

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.markdown("## STATUS MODEL")

        if is_ready:

            st.success(msg)

            st.markdown("### Total Audio")

            for cname in clf.class_names:

                count = clf.class_counts.get(
                    cname,
                    0
                )

                st.write(
                    f"{cname} : {count}"
                )

            st.markdown("---")

            st.markdown("### DEBUG")

            st.write(
                f"Total gagal: {len(clf.failed_files)}"
            )

            if clf.failed_files:

                with st.expander(
                    "Lihat File Gagal"
                ):

                    for item in clf.failed_files[:100]:

                        st.text(item)

        else:

            st.error(msg)

        st.markdown("---")

        if st.button(
            "🔄 Refresh ZIP",
            use_container_width=True
        ):

            st.cache_resource.clear()

            st.rerun()

    # ========================================================
    # NO DATA
    # ========================================================

    if not is_ready:

        st.warning(
            "Taruh ZIP di folder script"
        )

        return

    # ========================================================
    # UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload Audio",
        type=[
            'wav',
            'mp3',
            'm4a',
            'ogg',
            'flac',
            'opus'
        ]
    )

    if uploaded_file is not None:

        audio_bytes = uploaded_file.read()

        st.audio(audio_bytes)

        if st.button(
            "🔍 Analisis",
            type="primary"
        ):

            with st.spinner(
                "Menghitung..."
            ):

                try:

                    result = clf.predict(
                        audio_bytes
                    )

                    st.session_state[
                        'result'
                    ] = result

                except Exception as e:

                    st.error(str(e))

    # ========================================================
    # RESULT
    # ========================================================

    if 'result' in st.session_state:

        result = st.session_state['result']

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">
                {result['predicted_class']}
                </div>
                <div class="metric-label">
                Prediksi
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">
                {result['confidence']*100:.2f}%
                </div>
                <div class="metric-label">
                Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.plotly_chart(
            create_confidence_chart(
                result['ranked_predictions']
            ),
            use_container_width=True
        )

        export_data = {

            'timestamp': datetime.now().isoformat(),

            'prediction': result['predicted_class'],

            'confidence': result['confidence']
        }

        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(
                export_data,
                indent=2
            ),
            file_name="hasil_klasifikasi.json",
            mime="application/json"
        )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
