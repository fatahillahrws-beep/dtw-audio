import os
import sys
import zipfile
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional imports are installed by requirements.txt. The small fallback below helps local runs.
def ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

for pkg, imp in [
    ("librosa", "librosa"),
    ("scikit-learn", "sklearn"),
    ("scipy", "scipy"),
    ("pydub", "pydub"),
    ("imageio-ffmpeg", "imageio_ffmpeg"),
    ("kaggle", "kaggle"),
]:
    try:
        __import__(imp)
    except ImportError:
        pass

import librosa
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import pydub
    import imageio_ffmpeg
    pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pydub = None

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="DialectLab GMM",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".mp4"}
LOCAL_DATA_DIR = Path("kaggle_dataset")

# ==============================================================================
# MODERN UI STYLE
# ==============================================================================
def apply_modern_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

        :root {
            --bg0: #050816;
            --bg1: #08111f;
            --bg2: #0d1b2f;
            --glass: rgba(13, 27, 47, 0.72);
            --glass2: rgba(255,255,255,0.055);
            --line: rgba(148, 163, 184, 0.18);
            --line2: rgba(56, 189, 248, 0.35);
            --txt: #e5f0ff;
            --muted: #8aa4c5;
            --muted2: #5f7697;
            --cyan: #22d3ee;
            --sky: #38bdf8;
            --blue: #60a5fa;
            --violet: #a78bfa;
            --green: #34d399;
            --amber: #fbbf24;
            --rose: #fb7185;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(34,211,238,0.18), transparent 25%),
                radial-gradient(circle at 85% 20%, rgba(167,139,250,0.15), transparent 28%),
                radial-gradient(circle at 50% 90%, rgba(56,189,248,0.10), transparent 35%),
                linear-gradient(135deg, #050816 0%, #08111f 45%, #0b1020 100%);
            color: var(--txt);
        }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }

        section[data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.78) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

        .hero {
            position: relative;
            padding: 2.4rem 2.2rem;
            border: 1px solid var(--line);
            border-radius: 30px;
            background: linear-gradient(145deg, rgba(15,23,42,0.85), rgba(8,17,31,0.65));
            box-shadow: 0 24px 80px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
            overflow: hidden;
            margin-bottom: 1.4rem;
        }
        .hero:before {
            content: '';
            position: absolute;
            inset: -2px;
            background: radial-gradient(circle at 80% 10%, rgba(34,211,238,0.22), transparent 34%),
                        radial-gradient(circle at 5% 90%, rgba(167,139,250,0.18), transparent 35%);
            z-index: 0;
        }
        .hero > div { position: relative; z-index: 1; }
        .eyebrow {
            font-family: 'JetBrains Mono', monospace;
            color: var(--cyan);
            font-size: .72rem;
            letter-spacing: .23rem;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: .7rem;
        }
        .title {
            font-size: clamp(2.25rem, 5vw, 4.7rem);
            line-height: .95;
            letter-spacing: -0.08em;
            font-weight: 900;
            margin: 0;
        }
        .title span {
            background: linear-gradient(90deg, var(--cyan), var(--blue), var(--violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: var(--muted);
            max-width: 780px;
            margin-top: 1rem;
            font-size: 1rem;
            line-height: 1.7;
        }
        .badge-row { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.15rem; }
        .badge {
            font-family:'JetBrains Mono', monospace;
            font-size:.68rem;
            letter-spacing:.08rem;
            text-transform:uppercase;
            color:#cffafe;
            padding:.45rem .75rem;
            border-radius:999px;
            border:1px solid rgba(34,211,238,.25);
            background:rgba(34,211,238,.08);
        }

        .glass-card {
            border: 1px solid var(--line);
            background: linear-gradient(145deg, rgba(15,23,42,.72), rgba(13,27,47,.54));
            border-radius: 22px;
            padding: 1.2rem;
            box-shadow: 0 18px 50px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.05);
            margin-bottom: 1rem;
        }
        .mini-label {
            font-family:'JetBrains Mono', monospace;
            color:var(--muted2);
            text-transform:uppercase;
            letter-spacing:.15rem;
            font-size:.68rem;
            font-weight:700;
            margin-bottom:.35rem;
        }
        .metric-big {
            color:var(--txt);
            font-weight:900;
            font-size:clamp(1.35rem, 2.5vw, 2.3rem);
            letter-spacing:-.04em;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }
        .metric-accent {
            background: linear-gradient(90deg, var(--cyan), var(--blue));
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
        }
        .section-title {
            display:flex;
            align-items:center;
            gap:.75rem;
            margin:1.7rem 0 .8rem;
        }
        .section-no {
            font-family:'JetBrains Mono', monospace;
            font-size:.7rem;
            color:var(--cyan);
            border:1px solid rgba(34,211,238,.25);
            background:rgba(34,211,238,.08);
            border-radius:999px;
            padding:.35rem .55rem;
        }
        .section-text {
            font-weight:850;
            font-size:1rem;
            letter-spacing:-.02em;
            text-transform:uppercase;
        }
        .rank-line {
            display:flex;
            align-items:center;
            gap:.9rem;
            padding:.85rem 1rem;
            border:1px solid var(--line);
            border-radius:16px;
            background:rgba(15,23,42,.48);
            margin-bottom:.55rem;
        }
        .rank-line.top { border-color:rgba(34,211,238,.45); background:rgba(34,211,238,.07); }
        .rank-num { font-family:'JetBrains Mono', monospace; color:var(--muted2); width:2rem; text-align:right; }
        .rank-name { font-weight:800; min-width:110px; }
        .bar-bg { height:9px; background:rgba(148,163,184,.13); border-radius:999px; flex:1; overflow:hidden; }
        .bar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg, var(--blue), var(--cyan)); }
        .rank-score { font-family:'JetBrains Mono', monospace; color:var(--cyan); min-width:72px; text-align:right; font-weight:700; }

        .stButton > button {
            border-radius:14px !important;
            border:1px solid rgba(34,211,238,.35) !important;
            background:linear-gradient(135deg, rgba(34,211,238,.14), rgba(96,165,250,.10)) !important;
            color:#e0f2fe !important;
            font-family:'JetBrains Mono', monospace !important;
            text-transform:uppercase !important;
            letter-spacing:.08rem !important;
            font-weight:700 !important;
        }
        .stButton > button:hover { border-color:rgba(34,211,238,.7) !important; box-shadow:0 0 24px rgba(34,211,238,.12); }
        .stTabs [data-baseweb="tab-list"] { gap:.5rem; }
        .stTabs [data-baseweb="tab"] {
            border-radius:14px !important;
            background:rgba(15,23,42,.55) !important;
            border:1px solid var(--line) !important;
            color:var(--muted) !important;
            font-family:'JetBrains Mono', monospace !important;
            font-size:.72rem !important;
            letter-spacing:.08rem !important;
            text-transform:uppercase !important;
        }
        .stTabs [aria-selected="true"] {
            color:#cffafe !important;
            background:rgba(34,211,238,.10) !important;
            border-color:rgba(34,211,238,.35) !important;
        }
        div[data-testid="stFileUploader"] {
            border:1px dashed rgba(34,211,238,.35);
            border-radius:20px;
            padding:1rem;
            background:rgba(34,211,238,.045);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_modern_style()

# ==============================================================================
# KAGGLE DATA HANDLER
# ==============================================================================
def setup_kaggle_credentials():
    """Use Streamlit secrets first, then environment variables, then local kaggle.json."""
    try:
        if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
    except Exception:
        pass


def download_kaggle_dataset(dataset_slug: str, target_dir: str = "kaggle_dataset") -> Path:
    """Download and unzip a Kaggle dataset if target folder does not already contain audio or zip data."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    existing_audio = [p for p in target_path.rglob("*") if p.suffix.lower() in SUPPORTED_AUDIO]
    existing_zip = list(target_path.rglob("*.zip"))
    if existing_audio or existing_zip:
        return target_path

    if not dataset_slug or dataset_slug.strip() in {"username/nama-dataset", ""}:
        return target_path

    setup_kaggle_credentials()

    try:
        subprocess.check_call([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", dataset_slug,
            "-p", str(target_path),
            "--unzip",
        ])
    except Exception as e:
        st.error(
            "Gagal mengunduh dataset dari Kaggle. Pastikan KAGGLE_USERNAME, "
            "KAGGLE_KEY, dan dataset slug sudah benar.\n\n"
            f"Detail error: {e}"
        )

    return target_path


def extract_all_nested_zips(base_dir: Path) -> Path:
    """Extract ZIP files recursively. Keeps original zip files, extracts to folders with the same stem."""
    base_dir = Path(base_dir)
    extracted_marker = base_dir / ".extracted_ok"
    if extracted_marker.exists():
        return base_dir

    zip_files = list(base_dir.rglob("*.zip"))
    for z in zip_files:
        extract_dir = z.parent / z.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            st.warning(f"ZIP tidak bisa diekstrak: {z.name}. Error: {e}")

    try:
        extracted_marker.touch()
    except Exception:
        pass
    return base_dir


def find_dataset_root(base_dir: Path) -> Path:
    """
    Find a directory containing at least 2 label folders such as:
    Data_training/Logat Batak, Data_training/Logat Jawa, etc.
    """
    base_dir = Path(base_dir)
    candidates = []

    for folder in [base_dir] + [p for p in base_dir.rglob("*") if p.is_dir()]:
        subdirs = [p for p in folder.iterdir() if p.is_dir()]
        label_like = [p for p in subdirs if p.name.lower().startswith("logat")]
        audio_subdirs = []
        for p in subdirs:
            has_audio = any(a.suffix.lower() in SUPPORTED_AUDIO for a in p.rglob("*"))
            if has_audio:
                audio_subdirs.append(p)

        if len(label_like) >= 2:
            candidates.append((folder, len(label_like) + len(audio_subdirs)))
        elif len(audio_subdirs) >= 2:
            candidates.append((folder, len(audio_subdirs)))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return base_dir


def clean_label(folder_name: str) -> str:
    label = folder_name.strip()
    for prefix in ["Logat_", "Logat-", "Logat ", "logat_", "logat-", "logat "]:
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label.replace("_", " ").replace("-", " ").upper().strip()

# ==============================================================================
# AUDIO CORE: MFCC + DELTA + DELTA-DELTA
# ==============================================================================
class AcousticGMMCore:
    def __init__(self, sr=16000, n_mfcc=20, max_seconds=6, top_db=25):
        self.sr = int(sr)
        self.n_mfcc = int(n_mfcc)
        self.max_seconds = int(max_seconds)
        self.top_db = int(top_db)

    def load_audio(self, path: str):
        """Universal audio loader using pydub first, librosa fallback."""
        try:
            if pydub is not None:
                audio = pydub.AudioSegment.from_file(path).set_frame_rate(self.sr).set_channels(1)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                denom = float(1 << (8 * audio.sample_width - 1))
                y = samples / max(denom, 1.0)
                return y.astype(np.float32)
        except Exception:
            pass

        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True)
            return y.astype(np.float32)
        except Exception:
            return None

    def preprocess_audio(self, y):
        if y is None or len(y) == 0:
            return None

        max_len = self.sr * self.max_seconds
        if len(y) > max_len:
            y = y[:max_len]

        try:
            yt, _ = librosa.effects.trim(y, top_db=self.top_db)
            if len(yt) >= int(0.3 * self.sr):
                y = yt
        except Exception:
            pass

        peak = np.max(np.abs(y)) if len(y) else 0
        if peak > 0:
            y = y / (peak + 1e-8)
        return y.astype(np.float32)

    def extract_frame_features(self, y):
        """
        Output: frame-level matrix with shape (T, 3*n_mfcc).
        GMM is trained on frame distributions, not one vector per audio.
        """
        y = self.preprocess_audio(y)
        if y is None or len(y) < int(0.2 * self.sr):
            return None, None, None

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )

        # Cepstral mean and variance normalization per coefficient
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True) + 1e-8
        mfcc = mfcc / std

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.vstack([mfcc, delta, delta2]).T

        # Remove invalid rows
        feat = feat[np.all(np.isfinite(feat), axis=1)]
        if len(feat) == 0:
            return None, y, mfcc
        return feat.astype(np.float32), y, mfcc

    def spectral_centroid(self, y):
        try:
            sc = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
            return sc
        except Exception:
            return np.array([])

# ==============================================================================
# DATABASE + GMM MODEL
# ==============================================================================
def collect_label_audio_files(dataset_root: Path):
    dataset_root = Path(dataset_root)
    label_map = defaultdict(list)

    if not dataset_root.exists():
        return label_map

    # Preferred structure: root / Logat Batak / audio files
    for label_folder in dataset_root.iterdir():
        if not label_folder.is_dir():
            continue
        audio_files = [p for p in label_folder.rglob("*") if p.suffix.lower() in SUPPORTED_AUDIO]
        if audio_files:
            label_map[clean_label(label_folder.name)].extend(audio_files)

    # Fallback: audio files directly under nested folders, use parent folder as label.
    if not label_map:
        for audio_file in dataset_root.rglob("*"):
            if audio_file.suffix.lower() in SUPPORTED_AUDIO:
                label_map[clean_label(audio_file.parent.name)].append(audio_file)

    return label_map


@st.cache_resource(show_spinner=False)
def train_gmm_database(
    dataset_slug: str,
    use_kaggle: bool,
    local_dataset_path: str,
    sr: int,
    n_mfcc: int,
    max_seconds: int,
    n_components: int,
    covariance_type: str,
    random_state: int,
):
    core = AcousticGMMCore(sr=sr, n_mfcc=n_mfcc, max_seconds=max_seconds)

    if use_kaggle:
        base_path = download_kaggle_dataset(dataset_slug, str(LOCAL_DATA_DIR))
    else:
        base_path = Path(local_dataset_path) if local_dataset_path else Path(".")

    # Also support local zip in working directory, e.g. Data_training.zip
    if not use_kaggle and base_path.is_file() and base_path.suffix.lower() == ".zip":
        tmp_dir = Path("local_dataset_extracted")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(base_path, "r") as zf:
            zf.extractall(tmp_dir)
        base_path = tmp_dir

    base_path = extract_all_nested_zips(base_path)
    dataset_root = find_dataset_root(base_path)
    label_files = collect_label_audio_files(dataset_root)

    if len(label_files) < 2:
        return {
            "ok": False,
            "message": f"Dataset belum terbaca. Root yang dicari: {dataset_root}",
            "dataset_root": str(dataset_root),
            "models": {},
            "scaler": None,
            "waves": {},
            "mfcc_preview": {},
            "file_counts": {},
            "feature_dim": 0,
            "total_frames": 0,
        }

    raw_features_by_label = defaultdict(list)
    waves_by_label = defaultdict(list)
    mfcc_preview = {}
    file_counts = {}

    progress = st.progress(0, text="Membaca audio training...")
    all_files = [(label, f) for label, files in label_files.items() for f in files]
    total = max(len(all_files), 1)

    for i, (label, audio_file) in enumerate(all_files, start=1):
        y = core.load_audio(str(audio_file))
        feat, yt, mfcc = core.extract_frame_features(y)
        if feat is not None and len(feat) >= 5:
            raw_features_by_label[label].append(feat)
            if len(waves_by_label[label]) < 3:
                waves_by_label[label].append(yt)
            if label not in mfcc_preview and mfcc is not None:
                mfcc_preview[label] = mfcc
        progress.progress(i / total, text=f"Membaca training: {label} ({i}/{total})")
    progress.empty()

    labels = sorted(raw_features_by_label.keys())
    if len(labels) < 2:
        return {
            "ok": False,
            "message": "Audio berhasil ditemukan, tetapi label valid kurang dari 2.",
            "dataset_root": str(dataset_root),
            "models": {},
            "scaler": None,
            "waves": dict(waves_by_label),
            "mfcc_preview": mfcc_preview,
            "file_counts": {},
            "feature_dim": 0,
            "total_frames": 0,
        }

    file_counts = {label: len(raw_features_by_label[label]) for label in labels}
    all_train = np.vstack([np.vstack(raw_features_by_label[label]) for label in labels])

    scaler = StandardScaler()
    all_train_scaled = scaler.fit_transform(all_train)

    models = {}
    start_idx = 0
    for label in labels:
        x_label = np.vstack(raw_features_by_label[label])
        end_idx = start_idx + len(x_label)
        x_scaled = all_train_scaled[start_idx:end_idx]
        start_idx = end_idx

        # GMM components cannot exceed available samples.
        comp = min(int(n_components), max(1, len(x_scaled) // 10))
        comp = max(1, comp)

        gmm = GaussianMixture(
            n_components=comp,
            covariance_type=covariance_type,
            reg_covar=1e-5,
            max_iter=250,
            n_init=2,
            random_state=random_state,
        )
        gmm.fit(x_scaled)
        models[label] = gmm

    return {
        "ok": True,
        "message": "Model GMM berhasil dilatih.",
        "dataset_root": str(dataset_root),
        "models": models,
        "scaler": scaler,
        "waves": dict(waves_by_label),
        "mfcc_preview": mfcc_preview,
        "file_counts": file_counts,
        "feature_dim": int(all_train.shape[1]),
        "total_frames": int(all_train.shape[0]),
    }


def classify_with_gmm(test_features, models, scaler, temperature=1.0):
    x = scaler.transform(test_features)
    labels = list(models.keys())
    scores = []

    for label in labels:
        # Average log-likelihood per frame.
        ll = models[label].score(x)
        scores.append(ll)

    scores = np.array(scores, dtype=float)

    # Stable confidence. Temperature controls sharpness.
    temp = max(float(temperature), 1e-6)
    calibrated = (scores - np.max(scores)) / temp
    probs = softmax(calibrated)

    ranking = sorted(
        [(label, float(score), float(prob)) for label, score, prob in zip(labels, scores, probs)],
        key=lambda z: z[2],
        reverse=True,
    )
    return ranking

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_waveform(y, title="Waveform"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines", line=dict(width=1.2), fill="tozeroy"))
    fig.update_layout(
        title=title,
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=20),
        font=dict(color="#8aa4c5", family="JetBrains Mono"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def plot_mfcc_heatmap(mfcc, title="MFCC Heatmap"):
    fig = go.Figure(data=go.Heatmap(z=mfcc, colorscale="Turbo"))
    fig.update_layout(
        title=title,
        height=330,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=20),
        font=dict(color="#8aa4c5", family="JetBrains Mono"),
    )
    return fig


def plot_radar(ranking):
    labels = [x[0] for x in ranking]
    vals = [x[2] * 100 for x in ranking]
    if len(vals) == 0:
        vals = [0]
        labels = [""]
    fig = go.Figure(data=go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        line=dict(width=3),
    ))
    fig.update_layout(
        height=380,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(15,23,42,0.2)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.16)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        ),
        margin=dict(l=20, r=20, t=30, b=30),
        font=dict(color="#8aa4c5", family="JetBrains Mono"),
    )
    return fig


def plot_score_bar(ranking):
    labels = [r[0] for r in ranking][::-1]
    probs = [r[2] * 100 for r in ranking][::-1]
    fig = go.Figure(go.Bar(x=probs, y=labels, orientation="h", text=[f"{v:.1f}%" for v in probs], textposition="auto"))
    fig.update_layout(
        title="Confidence Ranking",
        height=max(300, 55 * len(labels)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=25),
        font=dict(color="#8aa4c5", family="JetBrains Mono"),
        xaxis=dict(range=[0, 100]),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def plot_spectral_centroid(sc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sc, mode="lines", line=dict(width=2), fill="tozeroy"))
    fig.update_layout(
        title="Spectral Centroid",
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=20),
        font=dict(color="#8aa4c5", family="JetBrains Mono"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def plot_pca_projection(test_features, database_result, winner):
    try:
        scaler = database_result["scaler"]
        models = database_result["models"]
        if scaler is None or not models:
            return None

        # Use GMM means as class centers, plus sampled test frames.
        class_points = []
        class_labels = []
        for label, gmm in models.items():
            means_original_space = scaler.inverse_transform(gmm.means_)
            class_points.append(means_original_space)
            class_labels.extend([label] * len(means_original_space))
        class_points = np.vstack(class_points)

        take = min(180, len(test_features))
        idx = np.linspace(0, len(test_features) - 1, take).astype(int)
        test_points = test_features[idx]

        all_points = np.vstack([class_points, test_points])
        pca = PCA(n_components=2, random_state=42)
        z = pca.fit_transform(all_points)
        z_class = z[: len(class_points)]
        z_test = z[len(class_points):]

        fig = go.Figure()
        unique_labels = sorted(set(class_labels))
        for lab in unique_labels:
            mask = np.array(class_labels) == lab
            fig.add_trace(go.Scatter(
                x=z_class[mask, 0],
                y=z_class[mask, 1],
                mode="markers+text",
                name=f"GMM {lab}",
                text=[lab] * np.sum(mask),
                textposition="top center",
                marker=dict(size=12, symbol="diamond", line=dict(width=1)),
            ))
        fig.add_trace(go.Scatter(
            x=z_test[:, 0],
            y=z_test[:, 1],
            mode="markers",
            name="Audio Uji",
            marker=dict(size=6, opacity=0.5),
        ))
        fig.update_layout(
            title=f"PCA Projection: Audio Uji vs GMM Centers ({winner})",
            height=420,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=45, b=20),
            font=dict(color="#8aa4c5", family="JetBrains Mono"),
        )
        fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
        fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
        return fig
    except Exception:
        return None

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="eyebrow">MFCC · GMM · Dialect Recognition</div>
                <h1 class="title">Dialect<span>Lab</span> GMM</h1>
                <div class="subtitle">
                    Sistem klasifikasi logat berbasis distribusi akustik. Audio diekstraksi menjadi MFCC, Delta, dan Delta-Delta, lalu setiap logat dimodelkan menggunakan Gaussian Mixture Model.
                </div>
                <div class="badge-row">
                    <span class="badge">MFCC</span>
                    <span class="badge">Delta</span>
                    <span class="badge">Delta-Delta</span>
                    <span class="badge">StandardScaler</span>
                    <span class="badge">Gaussian Mixture Model</span>
                    <span class="badge">Kaggle Dataset</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🎙️ DialectLab GMM")
        st.caption("Konfigurasi dataset dan model")

        use_kaggle = st.toggle("Ambil dataset dari Kaggle", value=True)
        dataset_slug = st.text_input(
            "Kaggle dataset slug",
            value="username/nama-dataset",
            help="Contoh: fatahillahrws/dialect-audio-training",
        )
        local_dataset_path = st.text_input(
            "Path dataset lokal / ZIP lokal",
            value="Data_training.zip",
            help="Dipakai kalau toggle Kaggle dimatikan. Bisa folder atau file ZIP.",
        )

        st.divider()
        st.markdown("#### Parameter Audio")
        sr = st.select_slider("Sampling rate", options=[8000, 16000, 22050, 44100], value=16000)
        n_mfcc = st.slider("Jumlah MFCC", 12, 40, 20)
        max_seconds = st.slider("Durasi maksimum audio", 3, 15, 6)

        st.divider()
        st.markdown("#### Parameter GMM")
        n_components = st.slider("Komponen GMM", 1, 16, 6)
        covariance_type = st.selectbox("Covariance type", ["diag", "full", "tied", "spherical"], index=0)
        temperature = st.slider("Kalibrasi confidence", 0.05, 5.0, 0.7, step=0.05)
        random_state = st.number_input("Random state", value=42, step=1)

        st.divider()
        if st.button("Reset cache dan latih ulang", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.caption("Struktur dataset: Data_training/Logat Batak, Logat Jawa, Logat Melayu, dst.")

    with st.spinner("Menyiapkan database dan melatih model GMM..."):
        database = train_gmm_database(
            dataset_slug=dataset_slug,
            use_kaggle=use_kaggle,
            local_dataset_path=local_dataset_path,
            sr=sr,
            n_mfcc=n_mfcc,
            max_seconds=max_seconds,
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=int(random_state),
        )

    if not database["ok"]:
        st.error(database["message"])
        st.info(
            "Pastikan dataset berbentuk folder seperti: Data_training/Logat Batak/audio.wav, "
            "Data_training/Logat Jawa/audio.wav, dan seterusnya. Jika memakai Kaggle, pastikan slug dan Secrets benar."
        )
        st.stop()

    labels = list(database["models"].keys())
    file_counts = database["file_counts"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="glass-card"><div class="mini-label">Jumlah Logat</div><div class="metric-big metric-accent">{len(labels)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="glass-card"><div class="mini-label">Total Audio</div><div class="metric-big">{sum(file_counts.values())}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="glass-card"><div class="mini-label">Frame Training</div><div class="metric-big">{database["total_frames"]:,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="glass-card"><div class="mini-label">Dimensi Fitur</div><div class="metric-big">{database["feature_dim"]}</div></div>', unsafe_allow_html=True)

    with st.expander("Lihat ringkasan database training"):
        st.write(f"**Dataset root terbaca:** `{database['dataset_root']}`")
        st.write("**Jumlah audio per logat:**")
        st.json(file_counts)

    st.markdown('<div class="section-title"><span class="section-no">INPUT</span><span class="section-text">Audio Uji</span></div>', unsafe_allow_html=True)
    tab_upload, tab_record = st.tabs(["Unggah Audio", "Rekam Langsung"])

    audio_bytes = None
    source_name = ""

    with tab_upload:
        uploaded = st.file_uploader(
            "Unggah file audio uji",
            type=[ext.replace(".", "") for ext in sorted(SUPPORTED_AUDIO)],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            audio_bytes = uploaded.read()
            source_name = uploaded.name

    with tab_record:
        recorded = st.audio_input("Rekam audio uji", label_visibility="collapsed")
        if recorded is not None:
            audio_bytes = recorded.read()
            source_name = "rekaman_langsung.wav"

    if audio_bytes is None:
        st.info("Unggah atau rekam audio untuk memulai klasifikasi logat.")
        st.stop()

    core = AcousticGMMCore(sr=sr, n_mfcc=n_mfcc, max_seconds=max_seconds)

    suffix = Path(source_name).suffix if Path(source_name).suffix else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    y_raw = core.load_audio(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    test_features, y_trimmed, mfcc = core.extract_frame_features(y_raw)
    if test_features is None:
        st.error("Audio uji gagal diproses. Pastikan file tidak rusak dan durasinya cukup.")
        st.stop()

    ranking = classify_with_gmm(
        test_features=test_features,
        models=database["models"],
        scaler=database["scaler"],
        temperature=temperature,
    )

    winner, winner_ll, winner_prob = ranking[0]
    second_prob = ranking[1][2] if len(ranking) > 1 else 0
    margin = max(0.0, winner_prob - second_prob)

    st.markdown('<div class="section-title"><span class="section-no">HASIL</span><span class="section-text">Prediksi GMM</span></div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f'<div class="glass-card"><div class="mini-label">Prediksi Logat</div><div class="metric-big metric-accent">{winner}</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="glass-card"><div class="mini-label">Confidence</div><div class="metric-big">{winner_prob*100:.1f}%</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="glass-card"><div class="mini-label">Margin vs Rank 2</div><div class="metric-big">{margin*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span class="section-no">01</span><span class="section-text">Peringkat Similarity</span></div>', unsafe_allow_html=True)
    for i, (label, ll, prob) in enumerate(ranking, start=1):
        top_cls = " top" if i == 1 else ""
        st.markdown(
            f"""
            <div class="rank-line{top_cls}">
                <div class="rank-num">#{i}</div>
                <div class="rank-name">{label}</div>
                <div class="bar-bg"><div class="bar-fill" style="width:{prob*100:.2f}%"></div></div>
                <div class="rank-score">{prob*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title"><span class="section-no">02</span><span class="section-text">Visualisasi Audio</span></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_waveform(y_trimmed, "Waveform Audio Uji"), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_mfcc_heatmap(mfcc, "MFCC Audio Uji"), use_container_width=True)

    st.markdown('<div class="section-title"><span class="section-no">03</span><span class="section-text">Distribusi Confidence</span></div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(plot_radar(ranking), use_container_width=True)
    with col_d:
        st.plotly_chart(plot_score_bar(ranking), use_container_width=True)

    st.markdown('<div class="section-title"><span class="section-no">04</span><span class="section-text">Analisis Spektral dan PCA</span></div>', unsafe_allow_html=True)
    col_e, col_f = st.columns(2)
    with col_e:
        sc = core.spectral_centroid(y_trimmed)
        st.plotly_chart(plot_spectral_centroid(sc), use_container_width=True)
    with col_f:
        fig_pca = plot_pca_projection(test_features, database, winner)
        if fig_pca is not None:
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("PCA projection belum dapat dibuat untuk data ini.")

    with st.expander("Penjelasan metode yang digunakan"):
        st.markdown(
            f"""
            **Pipeline analisis:**

            `Audio → Trim silence → Normalisasi amplitudo → MFCC → Delta → Delta-Delta → StandardScaler → GMM per logat → Log-likelihood → Softmax confidence`

            **Interpretasi hasil:**
            - Setiap logat dilatih sebagai distribusi Gaussian campuran.
            - Audio uji dihitung nilai **average log-likelihood** terhadap setiap model logat.
            - Logat dengan likelihood tertinggi dipilih sebagai prediksi.
            - Confidence adalah hasil kalibrasi softmax dari likelihood antar logat.

            **Model aktif:**
            - Sampling rate: `{sr}` Hz
            - MFCC: `{n_mfcc}`
            - Fitur per frame: `{database['feature_dim']}`
            - GMM components: `{n_components}`
            - Covariance type: `{covariance_type}`
            - Confidence temperature: `{temperature}`
            """
        )

if __name__ == "__main__":
    main()
