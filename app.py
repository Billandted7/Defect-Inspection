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

# =============================================
# DOWNLOAD MODEL WEIGHTS FROM HUGGING FACE
# =============================================
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
            exist_ok=True
        )

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

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Automated Visual Inspection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .pass-badge {
        background-color: #1b5e20;
        color: #69f0ae;
        padding: 20px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #69f0ae;
    }
    .fail-badge {
        background-color: #b71c1c;
        color: #ff8a80;
        padding: 20px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #ff8a80;
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
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None

THRESHOLD = 0.577

# =============================================
# SIDEBAR
# =============================================
st.sidebar.title("🔍 Visual Inspection System")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Inspect Component",
     "📊 Dashboard",
     "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status")

if st.session_state.model_trained:
    st.sidebar.success("✓ Model Ready")
    st.sidebar.markdown(
        "**Algorithm:** PatchCore  \n"
        "**Backbone:** WideResNet50  \n"
        "**Image AUROC:** 0.9976  \n"
        "**Training set:** 220 good parts  \n"
        "**Threshold:** 0.577 (calibrated)  \n"
        "**Good part pass rate:** 95%  \n"
        "**Defect detection rate:** 96%"
    )
else:
    st.sidebar.warning("⚠ Model not loaded")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built by Rebecca Moroney  \n"
    "Quality Engineering Portfolio"
)

# =============================================
# INFERENCE
# =============================================
def run_inference(img_rgb):
    os.environ["TRUST_REMOTE_CODE"] = "1"
    from anomalib.deploy import TorchInferencer

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
    elif defect_ratio > 0.20 and not is_edge:
        return "Flip / Orientation error", min(
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
        anomaly_map > 0.6
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        pred_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_img = img_rgb.copy()
    cv2.drawContours(
        contour_img, contours, -1,
        (255, 0, 0), 2
    )

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
                interpolation=cv2.INTER_LINEAR
            )
        else:
            zoom = contour_img
    else:
        zoom = contour_img

    return contour_img, zoom


# =============================================
# PAGE 1: INSPECT
# =============================================
if page == "🔍 Inspect Component":

    st.title("🔍 Automated Visual Inspection System")
    st.markdown(
        "Upload a component image for automated defect "
        "detection using PatchCore anomaly detection. "
        "Trained on 220 defect-free metal nut images — "
        "no defective examples required during training."
    )

    model_path = Path(
        "exported_model/weights/torch/model.pt")

    if not model_path.exists():
        st.error(
            "Exported model not found. "
            "Please wait for download to complete "
            "and refresh the page."
        )
        st.stop()

    if not st.session_state.model_trained:
        st.info(
            "Click below to initialise the system."
        )
        if st.button(
                "🚀 Initialise Inspection System",
                type="primary",
                use_container_width=True):
            with st.spinner("Initialising..."):
                time.sleep(1)
                st.session_state.model_trained = True
                st.rerun()
    else:
        st.success("✓ System ready for inspection")
        st.markdown("---")

        st.markdown("### Upload Component Image")
        uploaded = st.file_uploader(
            "Choose image (PNG or JPG)",
            type=["png", "jpg", "jpeg"],
            help=(
                "Upload a metal nut image. "
                "The model was trained on this "
                "component category specifically."
            )
        )

        if uploaded is not None:
            # Only run inference if this is a new image
            if uploaded.name != \
                    st.session_state.last_filename:

                file_bytes = np.frombuffer(
                    uploaded.read(), np.uint8)
                img_bgr = cv2.imdecode(
                    file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(
                    img_bgr, cv2.COLOR_BGR2RGB)

                with st.spinner(
                        "Running PatchCore "
                        "inference..."):
                    score, img_resized, anomaly_map\
                        = run_inference(img_rgb)

                verdict = "PASS" \
                    if score <= THRESHOLD else "FAIL"
                defect_type, confidence = \
                    classify_defect(anomaly_map, score)

                # Cache result to prevent rerun loop
                st.session_state.last_result = {
                    "score": score,
                    "img_resized": img_resized,
                    "anomaly_map": anomaly_map,
                    "verdict": verdict,
                    "defect_type": defect_type,
                    "confidence": confidence,
                    "filename": uploaded.name,
                }
                st.session_state.last_filename = \
                    uploaded.name

                # Log inspection
                st.session_state.inspection_log\
                    .append({
                        "timestamp": pd.Timestamp
                        .now().strftime(
                            "%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded.name,
                        "anomaly_score": round(
                            score, 4),
                        "threshold": THRESHOLD,
                        "verdict": verdict,
                        "defect_type": defect_type
                        if verdict == "FAIL"
                        else "—",
                    })

                pd.DataFrame(
                    st.session_state.inspection_log
                ).to_csv(
                    "inspection_log.csv", index=False)

            # Display cached result
            if st.session_state.last_result:
                r = st.session_state.last_result
                score = r["score"]
                img_resized = r["img_resized"]
                anomaly_map = r["anomaly_map"]
                verdict = r["verdict"]
                defect_type = r["defect_type"]
                confidence = r["confidence"]
                score_pct = score * 100
                threshold_pct = THRESHOLD * 100

                st.markdown("---")
                st.markdown("### Inspection Results")

                c1, c2, c3 = st.columns(3)

                with c1:
                    if verdict == "PASS":
                        st.markdown(
                            '<div class="pass-badge">'
                            '✓ PASS<br>'
                            'No Defect Detected'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="fail-badge">'
                            f'✗ FAIL<br>{defect_type}'
                            '</div>',
                            unsafe_allow_html=True
                        )

                with c2:
                    delta_val = (
                        score_pct - threshold_pct)
                    st.metric(
                        "Anomaly Score",
                        f"{score_pct:.1f}%",
                        delta=(
                            f"{delta_val:.1f}%"
                            " vs threshold"
                        )
                    )
                    if verdict == "FAIL":
                        st.markdown(
                            f"**Defect:** "
                            f"{defect_type}  \n"
                            f"**Confidence:** "
                            f"{confidence*100:.0f}%"
                        )

                with c3:
                    total = len(
                        st.session_state
                        .inspection_log)
                    passed = sum(
                        1 for r in
                        st.session_state
                        .inspection_log
                        if r["verdict"] == "PASS"
                    )
                    st.metric(
                        "Session Inspections", total)
                    st.metric(
                        "Pass Rate",
                        f"{passed/total*100:.0f}%"
                        if total > 0 else "—"
                    )

                st.markdown("---")

                if verdict == "PASS":
                    col1, col2, col3 = st.columns(
                        [1, 2, 1])
                    with col2:
                        st.markdown(
                            "**Component Image**")
                        st.image(
                            img_resized,
                            use_container_width=True
                        )
                        st.success(
                            "✓ Component passed. "
                            "No anomalies detected."
                        )
                else:
                    overlay = make_overlay(
                        img_resized, anomaly_map)
                    _, zoom = make_zoomed_mask(
                        img_resized, anomaly_map)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            "**Heatmap Overlay**")
                        st.image(
                            overlay,
                            use_container_width=True
                        )
                        st.caption(
                            "Red = high anomaly. "
                            "Blue = normal."
                        )

                    with col2:
                        st.markdown(
                            "**Defect Location "
                            "— Zoomed**")
                        st.image(
                            zoom,
                            use_container_width=True
                        )
                        st.caption(
                            "Red outline = predicted "
                            "defect region."
                        )

                    st.markdown("---")
                    st.error(
                        f"⚠️ Component failed. "
                        f"Detected: **{defect_type}**"
                        f". Do not ship."
                    )

                st.markdown("---")
                st.markdown("### Anomaly Score")
                st.progress(min(score, 1.0))
                st.markdown(
                    f"Score: **{score_pct:.1f}%** | "
                    f"Threshold: "
                    f"**{threshold_pct:.1f}%** | "
                    f"Verdict: **{verdict}**"
                )

# =============================================
# PAGE 2: DASHBOARD
# =============================================
elif page == "📊 Dashboard":

    st.title("📊 Inspection Dashboard")

    if os.path.exists("inspection_log.csv"):
        log_df = pd.read_csv("inspection_log.csv")
    else:
        log_df = pd.DataFrame(
            st.session_state.inspection_log)

    if len(log_df) == 0:
        st.info(
            "No inspections yet. "
            "Go to Inspect Component to get started."
        )
    else:
        total = len(log_df)
        passed = len(
            log_df[log_df["verdict"] == "PASS"])
        failed = total - passed
        pass_rate = passed / total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Inspected", total)
        c2.metric("Passed ✓", passed)
        c3.metric("Failed ✗", failed)
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
                    f"FAIL ({failed})"
                ],
                colors=["#4CAF50", "#FF5722"],
                autopct="%1.0f%%",
                textprops={"color": "white"}
            )
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
                linewidth=2
            )
            ax2.axhline(
                y=THRESHOLD * 100,
                color="red",
                linestyle="--",
                label=f"Threshold "
                      f"({THRESHOLD*100:.1f}%)"
            )
            ax2.set_xlabel(
                "Inspection #", color="white")
            ax2.set_ylabel(
                "Anomaly Score (%)", color="white")
            ax2.tick_params(colors="white")
            ax2.legend(
                facecolor="#1e2130",
                labelcolor="white"
            )
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        st.markdown("### Full Inspection Log")

        def colour_verdict(val):
            color = "#1b5e20" if val == "PASS" \
                else "#b71c1c"
            return f"background-color: {color}"

        st.dataframe(
            log_df.style.map(
                colour_verdict,
                subset=["verdict"]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Log (CSV)",
            log_df.to_csv(index=False),
            "inspection_log.csv",
            "text/csv",
            use_container_width=True
        )

# =============================================
# PAGE 3: ABOUT
# =============================================
elif page == "ℹ️ About":

    st.title("ℹ️ About This System")

    st.markdown("""
    ## What This Does

    Automated visual inspection of manufactured
    components using PatchCore anomaly detection.
    Detects scratches, bends, colour contamination,
    and orientation errors without ever training on
    defective parts.

    ---

    ## Model Performance

    | Metric | Score |
    |--------|-------|
    | Image AUROC | **0.9976** |
    | Pixel AUROC | **0.9868** |

    ---

    ## Threshold Calibration

    | | Value |
    |---|---|
    | Good parts — min score | 0.2948 |
    | Good parts — max score | 0.5952 |
    | Good parts — mean score | 0.4244 |
    | **Calibrated threshold** | **0.5770** |
    | Good parts passing | 95% (21/22) |
    | Defects caught | 96% (89/93) |

    ---

    ## How PatchCore Works

    1. WideResNet50 extracts patch-level features
       from every defect-free training image
    2. Features stored in a memory bank of what
       normal looks like
    3. At inspection time, each patch compared to
       nearest neighbours in memory bank
    4. Large distances flagged as anomalous
    5. Scores assembled into pixel-level heatmap

    ---

    ## Tech Stack

    Python · PyTorch · Anomalib 2.4.0 ·
    Streamlit · OpenCV · WideResNet50

    ---

    *Dataset: MVTec AD — Bergmann et al., CVPR 2019*
    """)

    if Path(
        "portfolio_outputs/portfolio_main.png"
    ).exists():
        st.markdown("---")
        st.markdown("## Example Results")
        st.image(
            "portfolio_outputs/portfolio_main.png",
            use_container_width=True
        )

    if Path(
        "portfolio_outputs/defect_deep_dive.png"
    ).exists():
        st.image(
            "portfolio_outputs/defect_deep_dive.png",
            use_container_width=True
        )