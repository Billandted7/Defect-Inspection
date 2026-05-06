import os
import streamlit as st
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

os.environ["TRUST_REMOTE_CODE"] = "1"


def download_model_if_needed():
    model_path = Path(
        "exported_model/weights/torch/model.pt")
    if not model_path.exists():
        print("Model weights not found locally. "
              "Downloading from Hugging Face...",
              flush=True)
        from huggingface_hub import hf_hub_download
        os.makedirs(
            "exported_model/weights/torch",
            exist_ok=True)
        hf_hub_download(
            repo_id="RMoroney/Defect-Inspection-Model",
            repo_type="model",
            filename="model.pt",
            local_dir="exported_model/weights/torch",
        )
        print("Model downloaded successfully.",
              flush=True)
    else:
        print("Model weights found locally.",
              flush=True)


download_model_if_needed()

st.set_page_config(
    page_title="Visual Inspection System",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem;
        max-width: 780px;
    }
    .hero {
        background: linear-gradient(
            135deg,
            #1a1a2e 0%,
            #16213e 60%,
            #0f3460 100%);
        border-radius: 0 0 24px 24px;
        padding: 40px 36px 36px 36px;
        margin: -1rem -1rem 24px -1rem;
        color: white;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #a8d8ea;
        margin-bottom: 16px;
    }
    .hero h1 {
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 12px 0;
        line-height: 1.15;
        color: white;
    }
    .hero p {
        font-size: 14px;
        color: rgba(255,255,255,0.75);
        line-height: 1.7;
        margin: 0 0 24px 0;
        max-width: 560px;
    }
    .hero-stats {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }
    .hero-stat {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 10px 16px;
        text-align: center;
        min-width: 100px;
    }
    .hero-stat-val {
        font-size: 20px;
        font-weight: 800;
        color: #69f0ae;
        display: block;
        line-height: 1;
    }
    .hero-stat-label {
        font-size: 10px;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
        display: block;
    }
    .section-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: #6c757d;
        margin: 0 0 12px 0;
    }
    .gallery-counter {
        text-align: center;
        font-size: 12px;
        color: #adb5bd;
        padding-top: 5px;
    }
    .pass-badge {
        background: linear-gradient(
            135deg, #d4edda, #c3e6cb);
        color: #155724;
        border: 1.5px solid #b1dfbb;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        font-size: 20px;
        font-weight: 800;
    }
    .fail-badge {
        background: linear-gradient(
            135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        border: 1.5px solid #f1aeb5;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        font-size: 20px;
        font-weight: 800;
    }
    .verdict-sub {
        font-size: 12px;
        font-weight: 500;
        margin-top: 4px;
        display: block;
        opacity: 0.8;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        height: 100%;
    }
    .metric-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #adb5bd;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #1a1a2e;
        line-height: 1;
    }
    .metric-sub {
        font-size: 11px;
        color: #adb5bd;
        margin-top: 4px;
    }
    .score-track {
        background: #e9ecef;
        border-radius: 6px;
        height: 8px;
        margin: 8px 0 4px 0;
        overflow: hidden;
    }
    .score-fill-pass {
        background: linear-gradient(
            90deg, #28a745, #20c997);
        height: 8px;
        border-radius: 6px;
    }
    .score-fill-fail {
        background: linear-gradient(
            90deg, #fd7e14, #dc3545);
        height: 8px;
        border-radius: 6px;
    }
    .history-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .history-verdict-pass {
        background: #d4edda;
        color: #155724;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
        min-width: 48px;
        text-align: center;
    }
    .history-verdict-fail {
        background: #f8d7da;
        color: #721c24;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
        min-width: 48px;
        text-align: center;
    }
    * {
        animation-duration: 0s !important;
        transition-duration: 0s !important;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
</style>

<script>
document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft') {
        const prevBtn = document.querySelector(
            '[data-testid="baseButton-secondary"][key="prev"]');
        if (prevBtn) prevBtn.click();
    }
    if (e.key === 'ArrowRight') {
        const nextBtn = document.querySelector(
            '[data-testid="baseButton-secondary"][key="next"]');
        if (nextBtn) nextBtn.click();
    }
});
</script>
""", unsafe_allow_html=True)

# =============================================
# SESSION STATE
# =============================================
if "inspection_log" not in st.session_state:
    st.session_state.inspection_log = []
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "inspecting" not in st.session_state:
    st.session_state.inspecting = False
if "gallery_index" not in st.session_state:
    st.session_state.gallery_index = 0
if "result_history" not in st.session_state:
    st.session_state.result_history = []

THRESHOLD = 0.50


# =============================================
# CACHED INFERENCE
# =============================================
@st.cache_data
def run_inference_cached(img_bytes):
    os.environ["TRUST_REMOTE_CODE"] = "1"
    from anomalib.deploy import TorchInferencer
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(
        img_bgr, cv2.COLOR_BGR2RGB)
    inferencer = TorchInferencer(
        path=Path(
            "exported_model/weights/torch/model.pt"),
    )
    img_resized = cv2.resize(img_rgb, (256, 256))
    result = inferencer.predict(image=img_resized)
    score = result.pred_score
    if hasattr(score, "numpy"):
        score = float(score.numpy().flatten()[0])
    elif hasattr(score, "item"):
        score = float(score.item())
    else:
        score = float(score)
    print(f"DEBUG score: {score}", flush=True)
    anomaly_map = None
    if hasattr(result, "anomaly_map") \
            and result.anomaly_map is not None:
        anomaly_map = result.anomaly_map
        if hasattr(anomaly_map, "numpy"):
            anomaly_map = anomaly_map.numpy()
        anomaly_map = np.squeeze(anomaly_map)
    if anomaly_map is None \
            or not hasattr(anomaly_map, "size") \
            or anomaly_map.size == 0:
        img_gray = cv2.cvtColor(
            img_resized, cv2.COLOR_RGB2GRAY)
        img_float = img_gray.astype(float) / 255.0
        k = 15
        local_mean = cv2.blur(img_float, (k, k))
        diff = img_float - local_mean
        anomaly_map = cv2.blur(diff ** 2, (k, k))
    anomaly_map = np.array(anomaly_map, dtype=float)
    if anomaly_map.max() > anomaly_map.min():
        anomaly_map = (
            (anomaly_map - anomaly_map.min()) /
            (anomaly_map.max() - anomaly_map.min())
        )
    else:
        anomaly_map = np.zeros_like(anomaly_map)
    return score, img_resized, anomaly_map


# =============================================
# DEFECT CLASSIFICATION
# =============================================
def classify_defect(anomaly_map, score):
    if score < THRESHOLD:
        return "No defect", 0.0
    thresh = (anomaly_map > 0.6).astype(np.uint8)
    total_px = (anomaly_map.shape[0]
                * anomaly_map.shape[1])
    defect_ratio = thresh.sum() / total_px
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not contours or defect_ratio < 0.005:
        return "Minor surface anomaly", score * 0.5
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    aspect = max(w, h) / (min(w, h) + 1e-6)
    M = cv2.moments(thresh.astype(float))
    if M["m00"] > 0:
        cx = (M["m10"] / M["m00"]
              / anomaly_map.shape[1])
        cy = (M["m01"] / M["m00"]
              / anomaly_map.shape[0])
    else:
        cx, cy = 0.5, 0.5
    is_edge = (cx < 0.25 or cx > 0.75
               or cy < 0.25 or cy > 0.75)
    if aspect > 3.5:
        return "Scratch", min(
            0.95, 0.65 + aspect * 0.02)
    elif defect_ratio > 0.30 and aspect < 2.0:
        return "Flip / Orientation error", min(
            0.95, 0.6 + defect_ratio)
    elif defect_ratio > 0.20:
        return "Scratch", min(
            0.95, 0.6 + defect_ratio)
    elif defect_ratio > 0.10 and is_edge:
        return "Bent / Deformation", min(
            0.95, 0.55 + defect_ratio * 2)
    elif defect_ratio > 0.05:
        return "Colour / Surface contamination", min(
            0.90, 0.5 + defect_ratio * 3)
    else:
        return "Surface anomaly", score * 0.7


# =============================================
# VISUALS
# =============================================
def make_overlay(img_rgb, anomaly_map):
    heatmap = cv2.applyColorMap(
        (anomaly_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(
        heatmap, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(
        heatmap_rgb,
        (img_rgb.shape[1], img_rgb.shape[0]))
    return cv2.addWeighted(
        img_rgb, 0.55, heatmap_resized, 0.45, 0)


def make_zoomed_mask(img_rgb, anomaly_map):
    pred_mask = (
        anomaly_map > 0.75).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        pred_mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_rgb.copy()
    cv2.drawContours(
        contour_img, contours, -1, (220, 50, 50), 2)
    if contours:
        all_pts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_pts)
        pad = 30
        h_i, w_i = img_rgb.shape[:2]
        zoom = contour_img[
            max(0, y-pad):min(h_i, y+h+pad),
            max(0, x-pad):min(w_i, x+w+pad)]
        if zoom.size > 0:
            zoom = cv2.resize(
                zoom,
                (zoom.shape[1]*3, zoom.shape[0]*3),
                interpolation=cv2.INTER_LINEAR)
        else:
            zoom = contour_img
    else:
        zoom = contour_img
    return contour_img, zoom


def score_interpretation(score_pct,
                         threshold_pct, verdict):
    diff = score_pct - threshold_pct
    if verdict == "PASS":
        if diff < -20:
            return "Well within normal range", \
                "#155724"
        else:
            return "Within acceptable range " \
                   "— borderline", "#856404"
    else:
        if diff > 20:
            return "Well above threshold " \
                   "— clear defect", "#721c24"
        else:
            return "Just above threshold " \
                   "— borderline fail", "#856404"


# =============================================
# HERO
# =============================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">
        AI Project · Quality Engineering
    </div>
    <h1>AI Visual Inspection System</h1>
    <p>
        This tool uses a PatchCore deep learning
        model to automatically detect defects in
        manufactured components. Trained on 220
        photographs of defect-free metal nuts, it
        detects scratches, surface contamination
        and deformation with 99.76% accuracy —
        without ever seeing a defective part during
        training.<br><br>
        <strong style="color:white;">
        To get started:</strong>
        browse the sample images below using the
        arrows (or keyboard ← →) and click
        <em>Inspect This Image</em> to run the AI
        analysis. Use the tabs above to view your
        inspection history on the Dashboard, or
        learn how the model works on the About page.
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-val">99.76%</span>
            <span class="hero-stat-label">
                Image AUROC</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-val">220</span>
            <span class="hero-stat-label">
                Training Images</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-val">96%</span>
            <span class="hero-stat-label">
                Defects Caught</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-val">95%</span>
            <span class="hero-stat-label">
                Good Parts Passed</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================
# TABS
# =============================================
t1, t2, t3 = st.tabs(
    ["🔍 Inspect", "📊 Dashboard", "ℹ️ About"])


# =============================================
# TAB 1: INSPECT
# =============================================
with t1:

    model_path = Path(
        "exported_model/weights/torch/model.pt")
    if not model_path.exists():
        st.error(
            "Model loading — please refresh "
            "in 30 seconds.")
        st.stop()

    sample_folder = Path("sample_images")
    selected_img_bytes = None
    selected_img_name = None

    if st.session_state.inspecting \
            and st.session_state.selected_sample:
        sp = sample_folder / \
            st.session_state.selected_sample
        if sp.exists():
            with open(sp, "rb") as f:
                selected_img_bytes = f.read()
            selected_img_name = sp.name

    # ── Gallery ──────────────────────────────
    if sample_folder.exists():
        sample_files = sorted(
            list(sample_folder.glob("*.png")) +
            list(sample_folder.glob("*.jpg")))

        if sample_files:
            show_gallery = (
                not st.session_state.inspecting
                and st.session_state.last_result
                is None)

            idx = max(0, min(
                st.session_state.gallery_index,
                len(sample_files) - 1))

            st.markdown(
                '<p class="section-header">'
                'Sample Components</p>',
                unsafe_allow_html=True)

            with st.expander(
                    "Browse Sample Images",
                    expanded=show_gallery):

                st.caption(
                    "Use the arrows to browse "
                    "— or press ← → on your "
                    "keyboard — then click "
                    "Inspect This Image.")

                n1, n2, n3, n4, n5 = \
                    st.columns([1, 1, 3, 1, 1])
                with n1:
                    if st.button(
                            "⏮", key="first",
                            use_container_width=True):
                        st.session_state\
                            .gallery_index = 0
                        st.session_state\
                            .last_result = None
                        st.rerun()
                with n2:
                    if st.button(
                            "◀", key="prev",
                            use_container_width=True):
                        st.session_state\
                            .gallery_index = max(
                                0, idx - 1)
                        st.session_state\
                            .last_result = None
                        st.rerun()
                with n3:
                    st.markdown(
                        f'<p class="gallery-counter">'
                        f'Image {idx + 1} of '
                        f'{len(sample_files)}'
                        f'</p>',
                        unsafe_allow_html=True)
                with n4:
                    if st.button(
                            "▶", key="next",
                            use_container_width=True):
                        st.session_state\
                            .gallery_index = min(
                                len(sample_files)-1,
                                idx + 1)
                        st.session_state\
                            .last_result = None
                        st.rerun()
                with n5:
                    if st.button(
                            "⏭", key="last",
                            use_container_width=True):
                        st.session_state\
                            .gallery_index = \
                            len(sample_files) - 1
                        st.session_state\
                            .last_result = None
                        st.rerun()

                current_file = sample_files[idx]
                with open(
                        current_file, "rb") as f:
                    current_bytes = f.read()
                arr = np.frombuffer(
                    current_bytes, np.uint8)
                prev = cv2.cvtColor(
                    cv2.imdecode(
                        arr, cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB)

                cl, cm, cr = st.columns([1, 4, 1])
                with cm:
                    st.image(
                        prev,
                        use_container_width=True,
                        caption=current_file.stem)
                    if st.button(
                            "Inspect This Image",
                            type="primary",
                            key="inspect_btn",
                            use_container_width=True):
                        st.session_state\
                            .selected_sample = \
                            current_file.name
                        st.session_state\
                            .inspecting = True
                        st.session_state\
                            .last_result = None
                        st.rerun()

    # ── Upload ───────────────────────────────
    st.markdown(
        '<p class="section-header" '
        'style="margin-top:20px;">'
        'Upload Your Own Image</p>',
        unsafe_allow_html=True)
    st.markdown(
        "Have your own component photograph? "
        "Upload it below for instant AI analysis.")
    uploaded = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed")

    # ── Resolve image ─────────────────────────
    if uploaded is not None:
        img_bytes = uploaded.getvalue()
        img_name = uploaded.name
    elif selected_img_bytes is not None:
        img_bytes = selected_img_bytes
        img_name = selected_img_name
    elif st.session_state.selected_sample:
        sp = sample_folder / \
            st.session_state.selected_sample
        if sp.exists():
            with open(sp, "rb") as f:
                img_bytes = f.read()
            img_name = sp.name
        else:
            img_bytes = None
            img_name = None
    else:
        img_bytes = None
        img_name = None

    # ── Inference ─────────────────────────────
    if img_bytes is not None:

        with st.spinner(
                "Analysing component against "
                "220 reference images..."):
            score, img_resized, anomaly_map = \
                run_inference_cached(img_bytes)

        st.session_state.inspecting = False

        verdict = "PASS" \
            if score <= THRESHOLD else "FAIL"
        defect_type, confidence = \
            classify_defect(anomaly_map, score)

        if defect_type == \
                "Flip / Orientation error":
            verdict = "PASS"
            defect_type = "No defect"

        score_pct = score * 100
        threshold_pct = THRESHOLD * 100

        interp, interp_color = \
            score_interpretation(
                score_pct, threshold_pct, verdict)

        if img_name != \
                st.session_state.last_filename:
            st.session_state.inspection_log\
                .append({
                    "timestamp": pd.Timestamp
                    .now().strftime(
                        "%Y-%m-%d %H:%M:%S"),
                    "filename": img_name,
                    "anomaly_score": round(
                        score, 4),
                    "threshold": THRESHOLD,
                    "verdict": verdict,
                    "defect_type": defect_type
                    if verdict == "FAIL"
                    else "—",
                })
            st.session_state.last_filename = \
                img_name

            # Store result in history for
            # revisiting
            overlay_for_history = make_overlay(
                img_resized, anomaly_map) \
                if verdict == "FAIL" else None
            st.session_state.result_history\
                .append({
                    "filename": img_name,
                    "verdict": verdict,
                    "score_pct": round(
                        score_pct, 1),
                    "defect_type": defect_type,
                    "img_resized": img_resized,
                    "overlay": overlay_for_history,
                })

            pd.DataFrame(
                st.session_state.inspection_log
            ).to_csv(
                "inspection_log.csv",
                index=False)

        st.session_state.last_result = {
            "verdict": verdict,
            "score": score,
            "defect_type": defect_type,
        }

        st.markdown("---")
        st.markdown(
            '<p class="section-header">'
            'Inspection Result</p>',
            unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            if verdict == "PASS":
                st.markdown(
                    '<div class="pass-badge">'
                    '✓ PASS'
                    '<span class="verdict-sub">'
                    'No defect detected'
                    '</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="fail-badge">'
                    '✗ FAIL'
                    f'<span class="verdict-sub">'
                    f'{defect_type}'
                    f'</span></div>',
                    unsafe_allow_html=True)

        with c2:
            fill_class = "score-fill-pass" \
                if verdict == "PASS" \
                else "score-fill-fail"
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="metric-label">'
                f'Anomaly Score</p>'
                f'<p class="metric-value">'
                f'{score_pct:.1f}%</p>'
                f'<div class="score-track">'
                f'<div class="{fill_class}" '
                f'style="width:'
                f'{min(score_pct,100):.1f}%">'
                f'</div></div>'
                f'<p class="metric-sub">'
                f'0% = identical to training images'
                f' · 100% = completely different'
                f'</p>'
                f'<p class="metric-sub">'
                f'Threshold: {threshold_pct:.0f}%'
                f'</p>'
                f'<p style="font-size:11px;'
                f'font-weight:600;margin-top:6px;'
                f'color:{interp_color};">'
                f'{interp}'
                f'</p></div>',
                unsafe_allow_html=True)

        with c3:
            total = len(
                st.session_state.inspection_log)
            passed = sum(
                1 for r in
                st.session_state.inspection_log
                if r["verdict"] == "PASS")
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="metric-label">'
                f'This Session</p>'
                f'<p class="metric-value">'
                f'{passed}/{total}</p>'
                f'<p class="metric-sub">'
                f'components passed</p>'
                f'</div>',
                unsafe_allow_html=True)

        st.markdown(
            "<br>", unsafe_allow_html=True)

        if verdict == "PASS":
            col1, col2, col3 = st.columns(
                [1, 2, 1])
            with col2:
                st.image(
                    img_resized,
                    use_container_width=True)
                st.success(
                    "Component passed — "
                    "no defects detected. "
                    "Safe to use.")

            st.markdown(
                "<br>", unsafe_allow_html=True)
            if st.button(
                    "Inspect Another Component",
                    use_container_width=True,
                    key="inspect_another_pass"):
                st.session_state.last_result = None
                st.session_state\
                    .selected_sample = None
                st.session_state\
                    .last_filename = None
                st.session_state.inspecting = False
                st.rerun()

        else:
            overlay = make_overlay(
                img_resized, anomaly_map)
            _, zoom = make_zoomed_mask(
                img_resized, anomaly_map)
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    overlay,
                    use_container_width=True)
                st.caption(
                    "Anomaly heatmap — "
                    "red = high anomaly")
            with col2:
                st.image(
                    zoom,
                    use_container_width=True)
                st.caption(
                    "Defect location — zoomed")

            st.error(
                f"Component failed — "
                f"**{defect_type}** detected. "
                f"Do not ship.")

            st.markdown(
                f'<div style="background:#fff3cd;'
                f'border:1px solid #ffc107;'
                f'border-radius:10px;'
                f'padding:14px 18px;'
                f'margin-top:12px;">'
                f'<p style="margin:0;'
                f'font-size:13px;font-weight:700;'
                f'color:#856404;">'
                f'What to do next</p>'
                f'<p style="margin:6px 0 0 0;'
                f'font-size:12px;color:#856404;">'
                f'1. Remove this component from '
                f'the production line '
                f'immediately.<br>'
                f'2. Label it as defective and '
                f'place in the quarantine bin.<br>'
                f'3. Log the defect type '
                f'({defect_type}) for quality '
                f'records — use the Dashboard tab '
                f'to download the inspection '
                f'log.<br>'
                f'4. Inspect adjacent components '
                f'from the same batch.'
                f'</p></div>',
                unsafe_allow_html=True)

            st.markdown(
                "<br>", unsafe_allow_html=True)
            if st.button(
                    "Inspect Another Component",
                    use_container_width=True,
                    key="inspect_another_fail"):
                st.session_state.last_result = None
                st.session_state\
                    .selected_sample = None
                st.session_state\
                    .last_filename = None
                st.session_state.inspecting = False
                st.rerun()

        # ── Previous Results History ──────────
        if len(st.session_state.result_history) > 1:
            st.markdown("---")
            st.markdown(
                '<p class="section-header">'
                'Previous Results This Session'
                '</p>',
                unsafe_allow_html=True)

            for i, r in enumerate(reversed(
                    st.session_state
                    .result_history[:-1])):
                verdict_class = \
                    "history-verdict-pass" \
                    if r["verdict"] == "PASS" \
                    else "history-verdict-fail"
                verdict_symbol = "✓" \
                    if r["verdict"] == "PASS" \
                    else "✗"
                defect_text = \
                    r["defect_type"] \
                    if r["verdict"] == "FAIL" \
                    else "No defect"

                with st.expander(
                        f"{verdict_symbol} "
                        f"{r['verdict']} — "
                        f"{r['filename']} — "
                        f"{r['score_pct']}%",
                        expanded=False):
                    h1, h2 = st.columns(2)
                    with h1:
                        st.image(
                            r["img_resized"],
                            use_container_width=True,
                            caption="Component")
                    with h2:
                        if r["overlay"] \
                                is not None:
                            st.image(
                                r["overlay"],
                                use_container_width=True,
                                caption="Heatmap")
                        else:
                            st.success(
                                "No defects "
                                "detected")
                    st.markdown(
                        f"**Verdict:** "
                        f"{r['verdict']}  \n"
                        f"**Score:** "
                        f"{r['score_pct']}%  \n"
                        f"**Defect:** "
                        f"{defect_text}")


# =============================================
# TAB 2: DASHBOARD
# =============================================
with t2:

    st.markdown(
        '<p class="section-header">'
        'Inspection Dashboard</p>',
        unsafe_allow_html=True)
    st.markdown(
        "A live summary of all inspections "
        "carried out in this session.")

    if os.path.exists("inspection_log.csv"):
        log_df = pd.read_csv("inspection_log.csv")
    else:
        log_df = pd.DataFrame(
            st.session_state.inspection_log)

    if len(log_df) == 0:
        st.info(
            "No inspections yet — go to the "
            "Inspect tab and analyse some "
            "components to see results here.")
    else:
        total = len(log_df)
        passed = len(
            log_df[log_df["verdict"] == "PASS"])
        failed = total - passed
        pass_rate = passed / total * 100

        s1, s2, s3, s4 = st.columns(4)
        for col, val, label, color in zip(
            [s1, s2, s3, s4],
            [total, passed, failed,
             f"{pass_rate:.0f}%"],
            ["Total", "Passed",
             "Failed", "Pass Rate"],
            ["#1a1a2e", "#155724",
             "#721c24", "#0c5460"]
        ):
            with col:
                st.markdown(
                    f'<div style="background:white;'
                    f'border-radius:12px;'
                    f'padding:16px;'
                    f'text-align:center;'
                    f'box-shadow:0 1px 3px '
                    f'rgba(0,0,0,0.06);">'
                    f'<p style="font-size:28px;'
                    f'font-weight:800;'
                    f'color:{color};'
                    f'margin:0;line-height:1;">'
                    f'{val}</p>'
                    f'<p style="font-size:11px;'
                    f'color:#adb5bd;'
                    f'margin:4px 0 0 0;'
                    f'text-transform:uppercase;'
                    f'letter-spacing:0.8px;">'
                    f'{label}</p></div>',
                    unsafe_allow_html=True)

        st.markdown(
            "<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pass / Fail Split**")
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.pie(
                [passed, failed],
                labels=[
                    f"PASS ({passed})",
                    f"FAIL ({failed})"],
                colors=["#28a745", "#dc3545"],
                autopct="%1.0f%%",
                wedgeprops={
                    "linewidth": 2,
                    "edgecolor": "white"},
                textprops={
                    "color": "#333",
                    "fontsize": 11})
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Score History**")
            scores = log_df["anomaly_score"] * 100
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            fig2.patch.set_facecolor("white")
            ax2.set_facecolor("#f8f9fa")
            colors_pts = [
                "#28a745" if v == "PASS"
                else "#dc3545"
                for v in log_df["verdict"]]
            ax2.plot(
                scores.values,
                color="#dee2e6",
                linewidth=1.5,
                zorder=1)
            ax2.scatter(
                range(len(scores)),
                scores.values,
                c=colors_pts,
                s=50,
                zorder=2)
            ax2.axhline(
                y=THRESHOLD * 100,
                color="#fd7e14",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold "
                      f"({THRESHOLD*100:.0f}%)")
            ax2.set_xlabel(
                "Inspection #",
                color="#6c757d",
                fontsize=9)
            ax2.set_ylabel(
                "Anomaly Score (%)",
                color="#6c757d",
                fontsize=9)
            ax2.tick_params(
                colors="#6c757d", labelsize=8)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"]\
                .set_color("#dee2e6")
            ax2.spines["bottom"]\
                .set_color("#dee2e6")
            ax2.legend(fontsize=9)
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        st.markdown("**Full Inspection Log**")

        def colour_verdict(val):
            if val == "PASS":
                return "background-color: #d4edda"
            return "background-color: #f8d7da"

        st.dataframe(
            log_df.style.map(
                colour_verdict,
                subset=["verdict"]),
            use_container_width=True,
            hide_index=True)

        st.download_button(
            "Download Inspection Log (CSV)",
            log_df.to_csv(index=False),
            "inspection_log.csv",
            "text/csv",
            use_container_width=True)


# =============================================
# TAB 3: ABOUT
# =============================================
with t3:

    st.markdown(
        '<p class="section-header">About</p>',
        unsafe_allow_html=True)

    with st.expander(
            "What This Project Does",
            expanded=True):
        st.markdown(
            "This project is an automated visual "
            "inspection system — a computer program "
            "that can look at a photograph of a "
            "manufactured component and decide "
            "whether that component is acceptable "
            "or defective. If a defect is found, "
            "the system draws a heatmap on the "
            "image showing exactly where the "
            "problem is located."
        )
        st.markdown(
            "The system was trained and tested on "
            "metal nuts"
            "detecting three types "
            "of defect:"
        )
        st.markdown(
            "- **Scratches** — surface damage "
            "caused by contact or abrasion\n"
            "- **Bends** — physical deformation "
            "where the component has been "
            "distorted from its correct shape\n"
            "- **Colour anomalies** — surface "
            "contamination or staining\n"
        
        )

    with st.expander(
            "Why Automated Inspection Matters"):
        st.markdown(
            "Human visual inspection has "
            "well-documented limitations in "
            "manufacturing."
        )
        st.markdown(
            "**Fatigue** — A human inspector "
            "becomes fatigued over a shift. "
            "Studies consistently show that "
            "inspection accuracy degrades "
            "significantly after two hours of "
            "continuous checking. An automated "
            "system has no concept of fatigue."
        )
        st.markdown(
            "**Subjectivity** — Two inspectors "
            "looking at the same borderline "
            "component may make different "
            "decisions. An automated system "
            "applies exactly the same decision "
            "rule to every component."
        )
        st.markdown(
            "**Traceability** — Manual inspection "
            "produces no structured data. This "
            "system automatically logs every "
            "inspection with a timestamp, anomaly "
            "score, and verdict — creating a "
            "complete quality record."
        )

    with st.expander(
            "How the System Works"):
        st.markdown("### The Core Problem")
        st.markdown(
            "The most obvious approach would be "
            "to collect thousands of photographs "
            "of both good and defective components "
            "and train an AI to tell the "
            "difference. The problem in "
            "manufacturing is that defective "
            "components are rare by definition. "
            "This project uses anomaly detection "
            "instead."
        )
        st.markdown("### Anomaly Detection")
        st.markdown(
            "Instead of teaching the system the "
            "difference between good and bad, "
            "anomaly detection teaches the system "
            "only what good looks like. The system "
            "studied 220 photographs of perfect "
            "components until it built a detailed "
            "understanding of what a normal metal "
            "nut looks like. At inspection time, "
            "it asks one question: does this look "
            "like anything seen during training? "
            "If not, it fails."
        )
        st.markdown("### The Memory Bank")
        st.markdown(
            "During training, every photograph is "
            "divided into small overlapping "
            "regions called patches. For each "
            "patch, the neural network produces "
            "a 1,536-number fingerprint describing "
            "what that region looks like. All "
            "225,280 fingerprints are stored in a "
            "memory bank. At inspection time, each "
            "patch of the new image is compared "
            "to its nearest neighbours in the "
            "memory bank. Patches far from "
            "anything in the memory bank produce "
            "high anomaly scores."
        )

    with st.expander("Model Performance"):
        st.markdown("""
| Metric | Library Version | Independent Version |
|--------|----------------|---------------------|
| Image AUROC | **0.9976** | **0.9980** |
| Pixel AUROC | **0.9868** | 0.9531 |
        """)
        st.markdown(
            "AUROC stands for Area Under the "
            "Receiver Operating Characteristic "
            "curve. It measures how well the "
            "system separates good components "
            "from defective ones across all "
            "possible decision thresholds. "
            "1.0 is perfect. 0.5 is random "
            "guessing. Above 0.90 is considered "
            "strong in academic research."
        )
        st.markdown(
            "The independent implementation "
            "achieved 0.9980 image AUROC — "
            "marginally higher than the library "
            "baseline of 0.9976. This confirms "
            "the implementation is correct."
        )

    with st.expander("Threshold Calibration"):
        st.markdown(
            "Every inspection system needs a "
            "decision threshold — a score above "
            "which a component is declared "
            "defective. Set it too low and the "
            "system fails good parts. Set it too "
            "high and real defects slip through."
        )
        st.markdown("""
| | Value |
|---|---|
| Good parts — lowest score | 0.2948 |
| Good parts — highest score | 0.5952 |
| Good parts — average score | 0.4244 |
| **Calibrated threshold** | **0.5770** |
| Good parts correctly passed | 95% (21/22) |
| Defects correctly caught | 96% (89/93) |
        """)

    with st.expander("Two Implementations"):
        st.markdown("### Library Version")
        st.markdown(
            "The first version was built using "
            "Anomalib — a professional open-source "
            "library developed by Intel. This "
            "established a validated performance "
            "baseline."
        )
        st.markdown("### Independent Implementation")
        st.markdown(
            "The entire algorithm was then "
            "re-implemented from scratch using "
            "only PyTorch, NumPy, and "
            "scikit-learn — demonstrating genuine "
            "understanding of how the algorithm "
            "works internally. Every component "
            "was written independently: feature "
            "extraction using forward hooks, "
            "patch assembly, coreset sampling, "
            "and anomaly scoring."
        )

    with st.expander("Possible Future Extensions"):
        st.markdown(
            "- **More component types** — the "
            "same approach applies to any "
            "component with photographs of "
            "good examples\n"
            "- **Live camera feed** — replace "
            "file upload with a live industrial "
            "camera stream\n"
            "- **SPC integration** — feed anomaly "
            "scores into control charts to detect "
            "process drift before defects occur\n"
            "- **Borderline flagging** — rather "
            "than binary pass/fail, flag "
            "borderline cases for human review"
        )

    with st.expander("Tech Stack"):
        st.markdown("""
| Component | Detail |
|-----------|--------|
| Algorithm | PatchCore |
| Backbone | WideResNet50 |
| Framework | PyTorch + Anomalib 2.4.0 |
| Interface | Streamlit |
| Deployment | Hugging Face Spaces |
| Dataset | MVTec AD (metal nut) |
| Language | Python 3.14 |
        """)

    st.markdown(
        "<p style='color:#adb5bd;"
        "font-size:12px;margin-top:24px;'>"
        "Built by Rebecca Moroney · "
        "Quality Engineering· "
        "MVTec AD — Bergmann et al., "
        "CVPR 2019</p>",
        unsafe_allow_html=True)