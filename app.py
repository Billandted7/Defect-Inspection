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
    page_title="Automated Visual Inspection System",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .pass-badge {
        background-color: #1b5e20;
        color: #69f0ae;
        padding: 24px 20px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #69f0ae;
        margin-bottom: 8px;
    }
    .fail-badge {
        background-color: #b71c1c;
        color: #ff8a80;
        padding: 24px 20px;
        border-radius: 10px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #ff8a80;
        margin-bottom: 8px;
    }
    .welcome-stat {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .welcome-stat h2 {
        font-size: 36px;
        font-weight: bold;
        color: #69f0ae;
        margin: 0;
    }
    .welcome-stat p {
        font-size: 13px;
        color: #aaaaaa;
        margin: 4px 0 0 0;
    }
    .step-box {
        background-color: #1e2130;
        border-left: 4px solid #2196F3;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    * {
        animation-duration: 0s !important;
        transition-duration: 0s !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# SESSION STATE
# =============================================
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
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
# SIDEBAR
# =============================================
st.sidebar.title("Visual Inspection System")
page = st.sidebar.radio(
    "Navigate",
    ["Inspect Component", "Dashboard", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")

if st.session_state.model_trained:
    st.sidebar.success("Model Ready")
    st.sidebar.markdown(
        "**Algorithm:** PatchCore  \n"
        "**Backbone:** WideResNet50  \n"
        "**Image AUROC:** 0.9976  \n"
        "**Training set:** 220 good parts  \n"
        "**Threshold:** 0.50 (calibrated)  \n"
        "**Good part pass rate:** 95%  \n"
        "**Defect detection rate:** 96%"
    )
else:
    st.sidebar.warning("Model not loaded")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by Rhys Moroney  \n"
    "Quality Engineering Portfolio  \n"
    "[GitHub](https://github.com/Billandted7/"
    "Defect-Inspection)"
)


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
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
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
        cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(
        heatmap, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(
        heatmap_rgb,
        (img_rgb.shape[1], img_rgb.shape[0])
    )
    overlay = cv2.addWeighted(
        img_rgb, 0.55, heatmap_resized, 0.45, 0)
    return overlay


def make_zoomed_mask(img_rgb, anomaly_map):
    pred_mask = (
        anomaly_map > 0.75).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        pred_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contour_img = img_rgb.copy()
    cv2.drawContours(
        contour_img, contours, -1, (255, 0, 0), 2)
    if contours:
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        pad = 30
        h_img, w_img = img_rgb.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        zoom = contour_img[y1:y2, x1:x2]
        if zoom.size > 0:
            zoom = cv2.resize(
                zoom,
                (zoom.shape[1] * 3,
                 zoom.shape[0] * 3),
                interpolation=cv2.INTER_LINEAR)
        else:
            zoom = contour_img
    else:
        zoom = contour_img
    return contour_img, zoom


# =============================================
# PAGE 1: INSPECT
# =============================================
if page == "Inspect Component":

    model_path = Path(
        "exported_model/weights/torch/model.pt")

    if not model_path.exists():
        st.error(
            "Model not found. "
            "Please wait and refresh the page.")
        st.stop()

    # =========================================
    # LANDING / WELCOME SCREEN
    # =========================================
    if not st.session_state.model_trained:

        st.markdown(
            "<h1 style='text-align:center;"
            "margin-bottom:4px;'>🔍 Automated Visual"
            " Inspection System</h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;"
            "color:#aaaaaa;font-size:16px;"
            "margin-bottom:32px;'>"
            "AI-powered defect detection for "
            "manufactured components</p>",
            unsafe_allow_html=True)

        # Key stats row
        s1, s2, s3, s4 = st.columns(4)
        stats = [
            ("99.76%", "Image AUROC"),
            ("220", "Training Images"),
            ("95%", "Good Part Pass Rate"),
            ("96%", "Defect Detection Rate"),
        ]
        for col, (val, label) in zip(
                [s1, s2, s3, s4], stats):
            with col:
                st.markdown(
                    f'<div class="welcome-stat">'
                    f'<h2>{val}</h2>'
                    f'<p>{label}</p>'
                    f'</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### How It Works")

        steps = [
            ("1", "Browse or upload a component image",
             "Use the sample gallery or upload your "
             "own photograph of a metal component."),
            ("2", "AI analyses the image",
             "PatchCore compares every patch of the "
             "image against 220 known-good reference "
             "images to detect any anomalies."),
            ("3", "Instant pass or fail verdict",
             "The system returns a PASS or FAIL "
             "verdict with a heatmap showing exactly "
             "where any defect is located."),
        ]

        for num, title, desc in steps:
            st.markdown(
                f'<div class="step-box">'
                f'<strong>Step {num} — {title}'
                f'</strong><br>'
                f'<span style="color:#aaaaaa;">'
                f'{desc}</span>'
                f'</div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### What Defects Can It Find?")

        d1, d2, d3, d4 = st.columns(4)
        defects = [
            ("🔧", "Scratches",
             "Surface marks from handling or machining"),
            ("📐", "Bent / Deformed",
             "Physical deformation at component edges"),
            ("🎨", "Colour Contamination",
             "Paint, oil, or surface staining"),
            ("⚠️", "Surface Anomalies",
             "Any other unexpected surface variation"),
        ]
        for col, (icon, name, desc) in zip(
                [d1, d2, d3, d4], defects):
            with col:
                st.markdown(
                    f'<div class="welcome-stat">'
                    f'<h2 style="font-size:28px;">'
                    f'{icon}</h2>'
                    f'<p style="color:white;'
                    f'font-weight:bold;">{name}</p>'
                    f'<p>{desc}</p>'
                    f'</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                    "Start Inspection System",
                    type="primary",
                    use_container_width=True):
                with st.spinner(
                        "Loading AI model..."):
                    time.sleep(1)
                    st.session_state\
                        .model_trained = True
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;"
            "color:#555555;font-size:12px;'>"
            "Trained on the MVTec AD dataset · "
            "PatchCore algorithm · "
            "WideResNet50 backbone</p>",
            unsafe_allow_html=True)

    # =========================================
    # MAIN INSPECTION INTERFACE
    # =========================================
    else:
        st.title("Component Inspection")
        st.markdown(
            "Browse the sample images below or "
            "upload your own. Click **Inspect This "
            "Image** to run the AI analysis."
        )
        st.markdown("---")

        # Sample image gallery
        sample_folder = Path("sample_images")
        selected_img_bytes = None
        selected_img_name = None

        if st.session_state.inspecting \
                and st.session_state.selected_sample:
            sp = Path("sample_images") / \
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
                    is None
                )

                idx = max(0, min(
                    st.session_state.gallery_index,
                    len(sample_files) - 1))

                with st.expander(
                        "Browse Sample Images",
                        expanded=show_gallery):

                    st.markdown(
                        "These are real metal nut "
                        "photographs from an "
                        "industrial inspection "
                        "dataset. Some have defects, "
                        "some are perfect — the AI "
                        "will tell you which."
                    )

                    n1, n2, n3, n4, n5 = \
                        st.columns([1, 1, 3, 1, 1])

                    with n1:
                        if st.button(
                                "<<", key="first",
                                use_container_width=True):
                            st.session_state\
                                .gallery_index = 0
                            st.session_state\
                                .last_result = None
                            st.rerun()
                    with n2:
                        if st.button(
                                "<", key="prev",
                                use_container_width=True):
                            st.session_state\
                                .gallery_index = max(
                                    0, idx - 1)
                            st.session_state\
                                .last_result = None
                            st.rerun()
                    with n3:
                        st.markdown(
                            f"<p style='text-align:"
                            f"center;padding-top:8px'>"
                            f"Image {idx + 1} of "
                            f"{len(sample_files)}"
                            f"</p>",
                            unsafe_allow_html=True)
                    with n4:
                        if st.button(
                                ">", key="next",
                                use_container_width=True):
                            st.session_state\
                                .gallery_index = min(
                                    len(sample_files)
                                    - 1, idx + 1)
                            st.session_state\
                                .last_result = None
                            st.rerun()
                    with n5:
                        if st.button(
                                ">>", key="last",
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

                    img_arr = np.frombuffer(
                        current_bytes, np.uint8)
                    img_bgr = cv2.imdecode(
                        img_arr, cv2.IMREAD_COLOR)
                    img_rgb_prev = cv2.cvtColor(
                        img_bgr, cv2.COLOR_BGR2RGB)

                    cl, cm, cr = st.columns(
                        [1, 4, 1])
                    with cm:
                        st.image(
                            img_rgb_prev,
                            use_container_width=True,
                            caption=current_file.stem)
                        if st.button(
                                "Inspect This Image",
                                type="primary",
                                key="inspect_gallery",
                                use_container_width=True):
                            st.session_state\
                                .selected_sample = \
                                current_file.name
                            st.session_state\
                                .inspecting = True
                            st.session_state\
                                .last_result = None
                            st.rerun()

        st.markdown("---")
        st.markdown("#### Or Upload Your Own Image")
        st.markdown(
            "Have your own metal nut photograph? "
            "Upload it below and the system will "
            "analyse it instantly."
        )
        uploaded = st.file_uploader(
            "Choose image (PNG or JPG)",
            type=["png", "jpg", "jpeg"],
            help=(
                "For best results upload a clear, "
                "well-lit photograph of a metal nut "
                "against a dark background, similar "
                "to the sample images above."
            )
        )

        # Determine image to process
        if uploaded is not None:
            img_bytes = uploaded.getvalue()
            img_name = uploaded.name
        elif selected_img_bytes is not None:
            img_bytes = selected_img_bytes
            img_name = selected_img_name
        elif st.session_state.selected_sample \
                is not None:
            sp = Path("sample_images") / \
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

        if img_bytes is not None:

            with st.spinner(
                    "Analysing image — "
                    "comparing against 220 "
                    "reference components..."):
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
            st.markdown("### Inspection Results")

            c1, c2, c3 = st.columns(3)

            with c1:
                if verdict == "PASS":
                    st.markdown(
                        '<div class="pass-badge">'
                        '✓ PASS<br>'
                        '<span style="font-size:14px;'
                        'font-weight:normal;">'
                        'No defect detected'
                        '</span></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="fail-badge">'
                        '✗ FAIL<br>'
                        f'<span style="font-size:14px;'
                        f'font-weight:normal;">'
                        f'{defect_type}'
                        f'</span></div>',
                        unsafe_allow_html=True)

            with c2:
                delta_val = score_pct - threshold_pct
                st.metric(
                    "Anomaly Score",
                    f"{score_pct:.1f}%",
                    delta=(
                        f"{delta_val:.1f}% "
                        f"vs threshold"),
                    help=(
                        "How different this component "
                        "looks compared to known-good "
                        "parts. Above 50% = FAIL."
                    )
                )
                if verdict == "FAIL":
                    st.markdown(
                        f"**Defect type:** "
                        f"{defect_type}  \n"
                        f"**Confidence:** "
                        f"{confidence*100:.0f}%")

            with c3:
                total = len(
                    st.session_state.inspection_log)
                passed = sum(
                    1 for r in
                    st.session_state.inspection_log
                    if r["verdict"] == "PASS")
                st.metric(
                    "Inspections This Session",
                    total)
                st.metric(
                    "Pass Rate",
                    f"{passed/total*100:.0f}%"
                    if total > 0 else "—")

            st.markdown("---")

            if verdict == "PASS":
                col1, col2, col3 = st.columns(
                    [1, 2, 1])
                with col2:
                    st.markdown("**Component Image**")
                    st.image(
                        img_resized,
                        use_container_width=True)
                    st.success(
                        "This component has passed "
                        "inspection. No defects were "
                        "detected. Safe to use.")
            else:
                overlay = make_overlay(
                    img_resized, anomaly_map)
                _, zoom = make_zoomed_mask(
                    img_resized, anomaly_map)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Anomaly Heatmap**")
                    st.image(
                        overlay,
                        use_container_width=True)
                    st.caption(
                        "Red areas = high anomaly "
                        "score. Blue = normal. The "
                        "AI is most concerned about "
                        "the red regions.")
                with col2:
                    st.markdown(
                        "**Defect Location**")
                    st.image(
                        zoom,
                        use_container_width=True)
                    st.caption(
                        "Red outline shows where "
                        "the predicted defect is "
                        "located on the component.")

                st.markdown("---")
                st.error(
                    f"⚠️ This component has "
                    f"**failed** inspection. "
                    f"Detected defect: "
                    f"**{defect_type}**. "
                    f"Do not use or ship this part.")

            st.markdown("---")
            st.markdown("### Anomaly Score")
            st.markdown(
                "The anomaly score measures how "
                "different this component looks "
                "compared to the 220 known-good "
                "training images. A score above "
                f"**{threshold_pct:.0f}%** triggers "
                "a FAIL verdict."
            )
            bar_color = "#4CAF50" \
                if verdict == "PASS" else "#FF5722"
            bar_html = (
                f'<div style="background:#333;'
                f'border-radius:4px;height:24px;'
                f'margin:8px 0;">'
                f'<div style="background:{bar_color};'
                f'width:{min(score_pct, 100):.1f}%;'
                f'height:24px;border-radius:4px;'
                f'display:flex;align-items:center;'
                f'padding-left:8px;color:white;'
                f'font-weight:bold;font-size:13px;">'
                f'{score_pct:.1f}%'
                f'</div></div>'
                f'<p style="color:#aaaaaa;'
                f'font-size:13px;">Threshold: '
                f'{threshold_pct:.0f}% | '
                f'Verdict: <strong>'
                f'{verdict}</strong></p>'
            )
            st.markdown(
                bar_html, unsafe_allow_html=True)


# =============================================
# PAGE 2: DASHBOARD
# =============================================
elif page == "Dashboard":

    st.title("Inspection Dashboard")
    st.markdown(
        "A summary of all inspections carried out "
        "in this session."
    )

    if os.path.exists("inspection_log.csv"):
        log_df = pd.read_csv("inspection_log.csv")
    else:
        log_df = pd.DataFrame(
            st.session_state.inspection_log)

    if len(log_df) == 0:
        st.info(
            "No inspections recorded yet. "
            "Go to **Inspect Component** and "
            "analyse some images to see results "
            "here.")
    else:
        total = len(log_df)
        passed = len(
            log_df[log_df["verdict"] == "PASS"])
        failed = total - passed
        pass_rate = passed / total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Inspected", total)
        c2.metric("Passed", passed)
        c3.metric("Failed", failed)
        c4.metric("Pass Rate", f"{pass_rate:.0f}%")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Pass / Fail Split")
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#1e2130")
            ax.set_facecolor("#1e2130")
            ax.pie(
                [passed, failed],
                labels=[
                    f"PASS ({passed})",
                    f"FAIL ({failed})"],
                colors=["#4CAF50", "#FF5722"],
                autopct="%1.0f%%",
                textprops={"color": "white"})
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### Score History")
            scores = log_df["anomaly_score"] * 100
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            fig2.patch.set_facecolor("#1e2130")
            ax2.set_facecolor("#1e2130")
            ax2.plot(
                scores.values,
                color="#2196F3",
                marker="o",
                linewidth=2)
            ax2.axhline(
                y=THRESHOLD * 100,
                color="red",
                linestyle="--",
                label=f"Threshold "
                      f"({THRESHOLD*100:.0f}%)")
            ax2.set_xlabel(
                "Inspection #", color="white")
            ax2.set_ylabel(
                "Anomaly Score (%)", color="white")
            ax2.tick_params(colors="white")
            ax2.legend(
                facecolor="#1e2130",
                labelcolor="white")
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        st.markdown("### Full Inspection Log")
        st.markdown(
            "Every inspection carried out this "
            "session. Download as CSV for records."
        )

        def colour_verdict(val):
            color = "#1b5e20" if val == "PASS" \
                else "#b71c1c"
            return f"background-color: {color}"

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
# PAGE 3: ABOUT
# =============================================
elif page == "About":

    st.title("About This System")
    st.markdown(
        "Technical documentation and background "
        "on the AI model powering this inspection "
        "system."
    )

    st.markdown("---")
    st.markdown("## What Problem Does This Solve?")
    st.markdown(
        "In manufacturing, a single defective "
        "component reaching a finished product can "
        "cause failure, recalls, or safety incidents. "
        "Traditional visual inspection relies on "
        "human operators who tire and make mistakes. "
        "This system provides consistent, "
        "automated inspection that never fatigues."
    )

    st.markdown("---")
    st.markdown("## Why PatchCore?")
    st.markdown(
        "Most defect detection systems require "
        "thousands of labelled examples of defective "
        "parts. In real factories, defects are rare "
        "by definition — you may only see a handful "
        "per year. PatchCore solves this by training "
        "only on good parts. It learns what normal "
        "looks like, then flags anything that "
        "deviates from that baseline."
    )

    st.markdown("---")
    st.markdown("## Model Performance")
    st.markdown("""
| Metric | Score | What It Means |
|--------|-------|---------------|
| Image AUROC | **0.9976** | Near-perfect separation of good vs defective |
| Pixel AUROC | **0.9868** | Accurate defect localisation |
| Image F1 Score | **0.9838** | Balance of precision and recall |
    """)

    st.markdown("---")
    st.markdown("## Threshold Calibration")
    st.markdown(
        "The acceptance threshold was set by running "
        "all known-good test images through the "
        "model and finding the score distribution."
    )
    st.markdown("""
| | Value |
|---|---|
| Good parts — lowest score | 0.2948 |
| Good parts — highest score | 0.5952 |
| Good parts — average score | 0.4244 |
| **Acceptance threshold** | **0.5000** |
| Good parts correctly passed | 95% |
| Defective parts correctly caught | 96% |
    """)

    st.markdown("---")
    st.markdown("## How PatchCore Works")
    st.markdown("""
1. **Feature extraction** — WideResNet50 processes
   every training image and extracts patch-level
   features from layers 2 and 3
2. **Memory bank** — features are stored as a
   compressed coreset representing normal appearance
3. **Inference** — at inspection time, each patch
   of the new image is compared to its nearest
   neighbours in the memory bank
4. **Scoring** — large distances from normal
   features produce high anomaly scores
5. **Localisation** — patch scores are assembled
   into a pixel-level heatmap showing defect location
    """)

    st.markdown("---")
    st.markdown("## Tech Stack")
    st.markdown("""
| Component | Technology |
|-----------|------------|
| Model | PatchCore (Anomalib 2.4.0) |
| Backbone | WideResNet50 |
| Framework | PyTorch |
| Interface | Streamlit |
| Image processing | OpenCV |
| Deployment | Hugging Face Spaces |
| Dataset | MVTec AD (metal nut category) |
    """)

    st.markdown("---")
    st.markdown(
        "<p style='color:#555555;font-size:12px;'>"
        "Dataset: MVTec AD — Bergmann et al., "
        "CVPR 2019. Built by Rhys Moroney as part "
        "of a Quality Engineering portfolio.</p>",
        unsafe_allow_html=True)