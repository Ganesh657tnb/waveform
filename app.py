import os
import tempfile
import subprocess
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Guardian OTT – Waveform Analyser",
    page_icon="🎵",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    /* Card panels */
    .plot-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(8px);
        margin-bottom: 24px;
    }

    /* Upload section */
    .upload-card {
        background: rgba(255,255,255,0.04);
        border: 1px dashed rgba(100,180,255,0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 12px;
    }

    /* Section titles */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #7ec8ff;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }

    /* Metric chip */
    .metric-chip {
        display: inline-block;
        background: rgba(126,200,255,0.15);
        border: 1px solid rgba(126,200,255,0.35);
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: #a8d8ff;
        margin: 4px 4px 4px 0;
    }

    /* Divider */
    .fancy-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 28px 0;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 32px 0 8px;
    }
    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7ec8ff, #a78bfa, #7ec8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Status badges */
    .badge-ok   { color: #4ade80; font-weight: 700; }
    .badge-wait { color: #facc15; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def extract_audio_from_video(video_path: str, wav_path: str) -> bool:
    """Use ffmpeg to extract mono 44100-Hz PCM WAV from a video file."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "44100",
            "-acodec", "pcm_s16le",
            wav_path,
        ],
        capture_output=True,
    )
    return result.returncode == 0 and os.path.exists(wav_path)


def load_wav(wav_path: str):
    """Read WAV → (sample_rate, float32 normalised samples)."""
    sr, data = wavfile.read(wav_path)
    # Handle stereo just in case
    if data.ndim > 1:
        data = data[:, 0]
    # Normalise to [-1, 1]
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    return sr, data


def compute_metrics(original: np.ndarray, watermarked: np.ndarray, sr: int) -> dict:
    """Compute basic audio quality metrics."""
    min_len = min(len(original), len(watermarked))
    orig = original[:min_len]
    wm   = watermarked[:min_len]

    diff = wm - orig
    noise_power  = np.mean(diff ** 2) + 1e-12
    signal_power = np.mean(orig ** 2) + 1e-12
    snr = 10 * np.log10(signal_power / noise_power)

    duration_orig = len(original) / sr
    duration_wm   = len(watermarked) / sr

    corr = float(np.corrcoef(orig, wm)[0, 1])

    return {
        "SNR (dB)"              : f"{snr:.2f}",
        "Correlation"           : f"{corr:.4f}",
        "Orig. Duration (s)"    : f"{duration_orig:.2f}",
        "WM Duration (s)"       : f"{duration_wm:.2f}",
        "Orig. Samples"         : f"{len(original):,}",
        "WM Samples"            : f"{len(watermarked):,}",
        "Sample Rate"           : f"{sr} Hz",
    }


# ─────────────────────────────────────────────
#  PLOT HELPERS  (all return matplotlib Figure)
# ─────────────────────────────────────────────
ZOOM = 4000          # samples to show in zoomed plots
FIG_BG   = "#0f0c29"
AX_BG    = "#1a1740"
COL_ORIG = "#38bdf8"   # sky-blue  → original
COL_WM   = "#f472b6"   # pink      → watermarked
COL_DIFF = "#a78bfa"   # purple    → difference

COMMON_PLOT_KW = dict(
    facecolor=FIG_BG,
)


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="#e2e8f0", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color="#94a3b8", fontsize=10)
    ax.set_ylabel(ylabel, color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(True, color="#1e293b", linewidth=0.8, linestyle="--")
    ax.legend(facecolor="#1e293b", edgecolor="#334155",
               labelcolor="#e2e8f0", fontsize=9)


def plot_single(samples: np.ndarray, sr: int, color: str, label: str, title: str):
    zoom = min(ZOOM, len(samples))
    t = np.linspace(0, zoom / sr, zoom)
    fig, ax = plt.subplots(figsize=(10, 3.2), **COMMON_PLOT_KW)
    fig.patch.set_facecolor(FIG_BG)
    ax.plot(t, samples[:zoom], color=color, linewidth=0.9, label=label)
    ax.fill_between(t, samples[:zoom], alpha=0.15, color=color)
    style_ax(ax, title, "Time (seconds)", "Amplitude")
    fig.tight_layout()
    return fig


def plot_comparison(original: np.ndarray, watermarked: np.ndarray, sr: int):
    zoom = min(ZOOM, len(original), len(watermarked))
    t = np.linspace(0, zoom / sr, zoom)

    fig = plt.figure(figsize=(12, 9), facecolor=FIG_BG)
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    # ── Row 1: Original ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, original[:zoom], color=COL_ORIG, linewidth=0.9, label="Original Audio")
    ax1.fill_between(t, original[:zoom], alpha=0.15, color=COL_ORIG)
    style_ax(ax1, "Original Audio Signal", "Time (s)", "Amplitude")

    # ── Row 2: Watermarked ───────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, watermarked[:zoom], color=COL_WM, linewidth=0.9, label="Watermarked Audio")
    ax2.fill_between(t, watermarked[:zoom], alpha=0.15, color=COL_WM)
    style_ax(ax2, "Watermarked Audio Signal", "Time (s)", "Amplitude")

    # ── Row 3: Overlay + Difference ──────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, original[:zoom],    color=COL_ORIG, linewidth=0.9,
             alpha=0.85, label="Original")
    ax3.plot(t, watermarked[:zoom], color=COL_WM,   linewidth=0.9,
             alpha=0.85, label="Watermarked")
    diff = watermarked[:zoom] - original[:zoom]
    ax3.fill_between(t, diff, alpha=0.30, color=COL_DIFF, label="Difference")
    style_ax(ax3, "Overlay Comparison + Difference", "Time (s)", "Amplitude")

    return fig


# ─────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🛡️ Guardian OTT · Waveform Analyser</h1>
  <p>Upload the <b>original</b> and <b>watermarked</b> videos to visualise and compare their audio signals.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Upload row ───────────────────────────────
col_orig, col_wm = st.columns(2, gap="large")

with col_orig:
    st.markdown('<p class="section-title">📁 Original Video</p>', unsafe_allow_html=True)
    orig_file = st.file_uploader(
        "Upload the original (non-watermarked) video",
        type=["mp4", "mkv", "avi", "mov"],
        key="orig",
        label_visibility="collapsed",
    )
    if orig_file:
        st.video(orig_file)
        st.markdown(
            f'<span class="badge-ok">✔ Loaded:</span> {orig_file.name} '
            f'({orig_file.size/1_000_000:.2f} MB)',
            unsafe_allow_html=True,
        )

with col_wm:
    st.markdown('<p class="section-title">🔒 Watermarked Video</p>', unsafe_allow_html=True)
    wm_file = st.file_uploader(
        "Upload the watermarked video",
        type=["mp4", "mkv", "avi", "mov"],
        key="wm",
        label_visibility="collapsed",
    )
    if wm_file:
        st.video(wm_file)
        st.markdown(
            f'<span class="badge-ok">✔ Loaded:</span> {wm_file.name} '
            f'({wm_file.size/1_000_000:.2f} MB)',
            unsafe_allow_html=True,
        )

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ── Analyse button ───────────────────────────
ready = orig_file is not None and wm_file is not None

if not ready:
    st.markdown(
        '<p class="badge-wait">⚠ Please upload both videos to enable analysis.</p>',
        unsafe_allow_html=True,
    )

analyse = st.button(
    "🔍  Analyse & Plot Waveforms",
    disabled=not ready,
    use_container_width=True,
)

# ── Processing ───────────────────────────────
if analyse and ready:

    with st.spinner("Extracting audio with FFmpeg — please wait…"):

        with tempfile.TemporaryDirectory() as tmp:

            # Save uploaded files to disk
            orig_video = os.path.join(tmp, "original.mp4")
            wm_video   = os.path.join(tmp, "watermarked.mp4")
            orig_wav   = os.path.join(tmp, "original.wav")
            wm_wav     = os.path.join(tmp, "watermarked.wav")

            with open(orig_video, "wb") as f:
                f.write(orig_file.getvalue())
            with open(wm_video, "wb") as f:
                f.write(wm_file.getvalue())

            # Extract audio
            ok1 = extract_audio_from_video(orig_video, orig_wav)
            ok2 = extract_audio_from_video(wm_video,   wm_wav)

            if not ok1:
                st.error("❌ Could not extract audio from the **Original** video. "
                         "Ensure FFmpeg is installed and the file has an audio track.")
                st.stop()
            if not ok2:
                st.error("❌ Could not extract audio from the **Watermarked** video. "
                         "Ensure FFmpeg is installed and the file has an audio track.")
                st.stop()

            # Load WAVs
            sr_orig, orig_samples = load_wav(orig_wav)
            sr_wm,   wm_samples   = load_wav(wm_wav)

            # Use the common sample rate for axis labels
            sr = sr_orig

            # Compute metrics
            metrics = compute_metrics(orig_samples, wm_samples, sr)

            # ── Metrics bar ──────────────────────────────
            st.markdown("### 📊 Audio Quality Metrics")
            chips_html = "".join(
                f'<span class="metric-chip"><b>{k}:</b> {v}</span>'
                for k, v in metrics.items()
            )
            st.markdown(f'<div class="plot-card">{chips_html}</div>',
                        unsafe_allow_html=True)

            st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

            # ── Plot 1: Original waveform ────────────────
            st.markdown(
                '<div class="plot-card">'
                '<p class="section-title">📈 Original Audio Signal Waveform</p>',
                unsafe_allow_html=True,
            )
            fig1 = plot_single(
                orig_samples, sr,
                color=COL_ORIG,
                label="Original Audio",
                title=f"Original Audio Signal — {orig_file.name}",
            )
            st.pyplot(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            plt.close(fig1)

            # ── Plot 2: Watermarked waveform ─────────────
            st.markdown(
                '<div class="plot-card">'
                '<p class="section-title">🔒 Watermarked Audio Signal Waveform</p>',
                unsafe_allow_html=True,
            )
            fig2 = plot_single(
                wm_samples, sr_wm,
                color=COL_WM,
                label="Watermarked Audio",
                title=f"Watermarked Audio Signal — {wm_file.name}",
            )
            st.pyplot(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            plt.close(fig2)

            # ── Plot 3: Comparison ───────────────────────
            st.markdown(
                '<div class="plot-card">'
                '<p class="section-title">🔀 Comparison: Original vs Watermarked</p>',
                unsafe_allow_html=True,
            )
            fig3 = plot_comparison(orig_samples, wm_samples, sr)
            st.pyplot(fig3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            plt.close(fig3)

            st.success("✅ Analysis complete! All three waveforms plotted above.")
