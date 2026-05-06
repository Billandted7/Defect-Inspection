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
    /* ── Global ── */
    .stApp {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ── Header ── */
    .site-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 1.5rem 0;
        border-bottom: 2px solid #e9ecef;
        margin-bottom: 1.5rem;
    }
    .site-title {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.3px;
        margin: 0;
    }
    .site-subtitle {
        font-size: 13px;
        color: #6c757d;
        margin: 2px 0 0 0;
    }
    .status-pill {
        background: #d4edda;
        color: #155724;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 600;
    }

    /* ── Cards ── */
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }
    .card-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #6c757d;
        margin-bottom: 8px;
    }

    /* ── Verdict badges ── */
    .pass-badge {
        background: #d4edda;
        color: #155724;
        border: 1.5px solid #c3e6cb;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 20px;
        font-weight: 700;
    }
    .fail-badge {
        background: #f8d7da;
        color: #721c24;
        border: 1.5px solid #f5c6cb;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 20px;
        font-weight: 700;
    }
    .verdict-sub {
        font-size: 12px;
        font-weight: 400;
        margin-top: 4px;
        display: block;
    }

    /* ── Score bar ── */
    .score-track {
        background: #e9ecef;
        border-radius: 6px;
        height: 10px;
        margin: 8px 0 4px 0;
        overflow: hidden;
    }
    .score-fill-pass {
        background: #28a745;
        height: 10px;
        border-radius: 6px;
    }
    .score-fill-fail {
        background: #dc3545;
        height: 10px;
        border-radius: 6px;
    }
    .score-label {
        font-size: 12px;
        color: #6c757d;
    }

    /* ── Stat chips ── */
    .stat-chip {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-value {
        font-size: 24px;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 11px;
        color: #6c757d;
        margin-top: 2px;
    }

    /* ── Gallery nav ── */
    .gallery-counter {
        text-align: center;
        font-size: 13px;
        color: #6c757d;
        padding-top: 6px;
    }

    /* ── Suppress animations ── */
    * {
        animation-duration: 0s !important;
        transition-duration: 0s !important;
    }

    /* ── Hide default streamlit footer ── */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
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
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
        cx = M["m10"] / M["m00"] / anomaly_map.shape[1]
        cy = M["m01"] / M["m00"] / anomaly_map.shape[0]
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


# =============================================
# NAVIGATION
# =============================================
nav1, nav2, nav3 = st.columns([3, 1, 1])
with nav1:
    st.markdown(
        '<p class="site-title">🔍 Visual '
        'Inspection System</p>'
        '<p class="site-subtitle">AI-powered '
        'defect detection · PatchCore · '
        'WideResNet50</p>',
        unsafe_allow_html=True)
with nav3:
    page = st.selectbox(
        "page",
        ["Inspect", "Dashboard", "About"],
        label_visibility="collapsed")


# =============================================
# PAGE: INSPECT
# =============================================
if page == "Inspect":

    model_path = Path(
        "exported_model/weights/torch/model.pt")
    if not model_path.exists():
        st.error(
            "Model loading — please wait "
            "30 seconds and refresh.")
        st.stop()

    # ── Gallery ──────────────────────────────
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

            with st.expander(
                    "Sample Images — click any to inspect",
                    expanded=show_gallery):

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
                        f'{len(sample_files)}</p>',
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
                with open(current_file, "rb") as f:
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
        '<p style="font-size:13px;color:#6c757d;'
        'margin:16px 0 4px 0;">— or upload your '
        'own image —</p>',
        unsafe_allow_html=True)
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

    # ── Run inference ─────────────────────────
    if img_bytes is not None:

        with st.spinner(
                "Analysing component..."):
            score, img_resized, anomaly_map = \
                run_inference_cached(img_bytes)

        st.session_state.inspecting = False

        verdict = "PASS" \
            if score <= THRESHOLD else "FAIL"
        defect_type, confidence = \
            classify_defect(anomaly_map, score)

        if defect_type == "Flip / Orientation error":
            verdict = "PASS"
            defect_type = "No defect"

        score_pct = score * 100
        threshold_pct = THRESHOLD * 100

        if img_name != \
                st.session_state.last_filename:
            st.session_state.inspection_log.append({
                "timestamp": pd.Timestamp.now()
                .strftime("%Y-%m-%d %H:%M:%S"),
                "filename": img_name,
                "anomaly_score": round(score, 4),
                "threshold": THRESHOLD,
                "verdict": verdict,
                "defect_type": defect_type
                if verdict == "FAIL" else "—",
            })
            st.session_state.last_filename = img_name
            pd.DataFrame(
                st.session_state.inspection_log
            ).to_csv(
                "inspection_log.csv", index=False)

        st.session_state.last_result = {
            "verdict": verdict,
            "score": score,
            "defect_type": defect_type,
        }

        # ── Results ───────────────────────────
        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        with c1:
            if verdict == "PASS":
                st.markdown(
                    '<div class="pass-badge">'
                    '✓ PASS'
                    '<span class="verdict-sub">'
                    'No defect detected</span>'
                    '</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="fail-badge">'
                    '✗ FAIL'
                    f'<span class="verdict-sub">'
                    f'{defect_type}</span>'
                    '</div>',
                    unsafe_allow_html=True)

        with c2:
            st.markdown(
                '<div class="card">'
                '<p class="card-label">'
                'Anomaly Score</p>'
                f'<p style="font-size:28px;'
                f'font-weight:700;margin:0;'
                f'color:#1a1a2e;">'
                f'{score_pct:.1f}%</p>'
                f'<div class="score-track">'
                f'<div class="score-fill-'
                f'{"pass" if verdict == "PASS" else "fail"}" '
                f'style="width:{min(score_pct,100):.1f}%">'
                f'</div></div>'
                f'<p class="score-label">'
                f'Threshold: {threshold_pct:.0f}%</p>'
                f'</div>',
                unsafe_allow_html=True)

        with c3:
            total = len(
                st.session_state.inspection_log)
            passed = sum(
                1 for r in
                st.session_state.inspection_log
                if r["verdict"] == "PASS")
            st.markdown(
                '<div class="card">'
                '<p class="card-label">'
                'This Session</p>'
                f'<p style="font-size:28px;'
                f'font-weight:700;margin:0;'
                f'color:#1a1a2e;">'
                f'{passed}/{total}</p>'
                f'<p class="score-label">'
                f'components passed</p>'
                f'</div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if verdict == "PASS":
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    img_resized,
                    use_container_width=True)
                st.success(
                    "Component passed — no defects.")
        else:
            overlay = make_overlay(
                img_resized, anomaly_map)
            _, zoom = make_zoomed_mask(
                img_resized, anomaly_map)
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    overlay,
                    use_container_width=True,
                    caption="Anomaly heatmap — "
                            "red = high anomaly")
            with col2:
                st.image(
                    zoom,
                    use_container_width=True,
                    caption="Defect location — zoomed")
            st.error(
                f"Failed — **{defect_type}** "
                f"detected. Do not ship.")

        if verdict == "FAIL" and defect_type:
            st.markdown(
                '<div class="card" style="'
                'margin-top:16px;">'
                '<p class="card-label">Defect Detail'
                '</p>'
                f'<p style="font-size:15px;'
                f'color:#1a1a2e;margin:0;">'
                f'<strong>Type:</strong> '
                f'{defect_type}</p>'
                f'<p style="font-size:15px;'
                f'color:#1a1a2e;margin:4px 0 0 0;">'
                f'<strong>Confidence:</strong> '
                f'{confidence*100:.0f}%</p>'
                f'</div>',
                unsafe_allow_html=True)


# =============================================
# PAGE: DASHBOARD
# =============================================
elif page == "Dashboard":

    st.markdown("## Inspection Dashboard")

    if os.path.exists("inspection_log.csv"):
        log_df = pd.read_csv("inspection_log.csv")
    else:
        log_df = pd.DataFrame(
            st.session_state.inspection_log)

    if len(log_df) == 0:
        st.info(
            "No inspections yet — go to "
            "Inspect to get started.")
    else:
        total = len(log_df)
        passed = len(
            log_df[log_df["verdict"] == "PASS"])
        failed = total - passed
        pass_rate = passed / total * 100

        s1, s2, s3, s4 = st.columns(4)
        for col, val, label in zip(
            [s1, s2, s3, s4],
            [total, passed, failed,
             f"{pass_rate:.0f}%"],
            ["Total", "Passed",
             "Failed", "Pass Rate"]
        ):
            with col:
                st.markdown(
                    f'<div class="stat-chip">'
                    f'<p class="stat-value">'
                    f'{val}</p>'
                    f'<p class="stat-label">'
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
                textprops={"color": "#333"})
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Score History**")
            scores = log_df["anomaly_score"] * 100
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            fig2.patch.set_facecolor("white")
            ax2.set_facecolor("#f8f9fa")
            ax2.plot(
                scores.values,
                color="#1a1a2e",
                marker="o",
                linewidth=2,
                markersize=5)
            ax2.axhline(
                y=THRESHOLD * 100,
                color="#dc3545",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold "
                      f"({THRESHOLD*100:.0f}%)")
            ax2.set_xlabel(
                "Inspection #",
                color="#6c757d",
                fontsize=10)
            ax2.set_ylabel(
                "Anomaly Score (%)",
                color="#6c757d",
                fontsize=10)
            ax2.tick_params(
                colors="#6c757d", labelsize=9)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.legend(fontsize=9)
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        st.markdown("**Inspection Log**")

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
            "Download CSV",
            log_df.to_csv(index=False),
            "inspection_log.csv",
            "text/csv",
            use_container_width=True)


# =============================================
# PAGE: ABOUT
# =============================================
elif page == "About":

    st.markdown("## About This System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Performance")
        st.markdown("""
| Metric | Score |
|--------|-------|
| Image AUROC | **0.9976** |
| Pixel AUROC | **0.9868** |
| F1 Score | **0.9838** |
| Good parts passed | **95%** |
| Defects caught | **96%** |
        """)

    with col2:
        st.markdown("### Tech Stack")
        st.markdown("""
| Component | Detail |
|-----------|--------|
| Algorithm | PatchCore |
| Backbone | WideResNet50 |
| Framework | PyTorch + Anomalib |
| Interface | Streamlit |
| Deployment | Hugging Face Spaces |
| Dataset | MVTec AD |
        """)

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown(
        "PatchCore trains only on defect-free "
        "images — 220 in this case. It builds a "
        "memory bank of what a good component looks "
        "like at the patch level. At inspection "
        "time, every patch in the new image is "
        "compared to its nearest neighbours in the "
        "memory bank. Patches that are far from any "
        "known-good patch produce high anomaly "
        "scores and are flagged as defective."
    )

    st.markdown("---")
    st.markdown("### Why This Matters")
    st.markdown(
        "In real manufacturing, defective parts "
        "are rare by definition. You cannot collect "
        "thousands of defect examples to train a "
        "classifier. PatchCore solves this — it "
        "needs only good parts to train, matching "
        "real factory conditions exactly."
    )

    st.markdown(
        "<p style='color:#aaaaaa;font-size:12px;"
        "margin-top:32px;'>"
        "Built by Rhys Moroney · "
        "Quality Engineering Portfolio · "
        "MVTec AD dataset — Bergmann et al., "
        "CVPR 2019</p>",
        unsafe_allow_html=True)