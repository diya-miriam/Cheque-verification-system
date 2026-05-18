"""
ui/app6.py — Cheque Signature Verification UI (API-integrated)

Verification, feedback, and retraining are all handled by the FastAPI backend (api_new.py).
The UI calls POST /verify (returns ROI as base64), then POST /feedback with the session_id.

Start the API before running this UI:
    python -m uvicorn api_new:app --reload --port 8000
Then run:
    python -m streamlit run ui/app6.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import retraining_cfg

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE             = "http://127.0.0.1:8000"
THRESHOLD            = 0.8
MIN_FEEDBACK_SAMPLES = retraining_cfg.training.df_length

# ── Session state ──────────────────────────────────────────────────────────────
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "model_margin_1.0.pt"
if "result" not in st.session_state:
    st.session_state.result = None
if "writer_id" not in st.session_state:
    st.session_state.writer_id = ""

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cheque Signature Verification",
    page_icon="🧾",
    layout="wide",
)

# ── API helpers ────────────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 503


def check_api_online() -> bool:
    _, code = api_get("/health")
    return code == 200


def get_available_models_from_api() -> list[str]:
    resp, code = api_get("/models")
    if code == 200:
        return resp.get("models", [])
    # Fallback: read from disk directly
    models_dir = PROJECT_ROOT / "src" / "models" / "checkpoints"
    return sorted([f.name for f in models_dir.glob("*.pt")])


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Model Control")

api_online = check_api_online()
if api_online:
    st.sidebar.success("🟢 API connected")
else:
    st.sidebar.error("🔴 API offline")
    st.sidebar.code("python -m uvicorn api_new:app --reload --port 8000")

mode = st.sidebar.radio("Select Mode", ["🔍 Inference", "📊 Dashboard"])

available_models = get_available_models_from_api()

if st.session_state.selected_model not in available_models:
    st.session_state.selected_model = available_models[0] if available_models else "model_margin_1.0.pt"

selected_model = st.sidebar.selectbox(
    "Select Model Version",
    available_models,
    index=available_models.index(st.session_state.selected_model),
)

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.result = None
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE PAGE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "🔍 Inference":

    st.title("🧾 Cheque Signature Verification")
    st.caption(
        "The scanned cheque image is processed using your preprocessing pipeline to extract the signature. "
        "The reference signature is prepared using the same final image formatting used during training."
    )
    st.divider()

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("📤 Upload Inputs")

        with st.container(border=True):
            account_number = st.text_input(
                "User Account Number",
                help="Used to ensure proper data splitting during retraining.",
            )
            st.session_state.writer_id = account_number

            cheque_file = st.file_uploader(
                "Scanned Cheque Image",
                type=["png", "jpg", "jpeg"],
                help="Full cheque image — pipeline extracts the signature region automatically.",
            )
            ref_file = st.file_uploader(
                "Reference Signature",
                type=["png", "jpg", "jpeg"],
                help="Upload a cropped reference signature image.",
            )

            verify_btn = st.button(
                "✅ Verify Signature",
                use_container_width=True,
                disabled=not (cheque_file and ref_file and account_number and api_online),
            )

            if not api_online:
                st.warning("Start the API server to enable verification.")
            elif not (cheque_file and ref_file):
                st.info("Upload both the cheque image and the reference signature to enable verification.")

        # ── Image previews (exactly like original app4.py) ─────────────────
        if cheque_file:
            st.image(Image.open(cheque_file).convert("RGB"),
                     caption="Uploaded Scanned Cheque", use_container_width=True)
        if ref_file:
            st.image(Image.open(ref_file).convert("RGB"),
                     caption="Uploaded Reference Signature", use_container_width=True)

    with right:
        st.subheader("📌 Result")

        with st.container(border=True):
            if st.session_state.result is None:
                st.write("Upload both images and click **Verify Signature**.")
                st.caption("The extracted signature from the cheque and final result will appear here.")
            else:
                r     = st.session_state.result
                label = r["label"]

                if label == "GENUINE":
                    st.success("✅ GENUINE")
                else:
                    st.error("⚠️ FORGED")

                st.metric("Euclidean Distance", f"{r['distance']:.4f}")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(r["roi"], caption="Extracted Signature (from Cheque Pipeline)", use_container_width=True)
                with c2:
                    st.image(r["ref"], caption="Reference Signature", use_container_width=True)

        st.caption(f"Decision rule: distance < {THRESHOLD:.2f} → GENUINE, otherwise FORGED.")
        st.caption(f"Model used: {st.session_state.selected_model}")

    # ── Verify button handler ──────────────────────────────────────────────────
    if verify_btn:
        with st.spinner("Sending to API and verifying signature..."):
            try:
                cheque_bytes = cheque_file.getvalue()
                ref_bytes    = ref_file.getvalue()

                # Call the API for verification
                files = {
                    "cheque":    (cheque_file.name, cheque_bytes, "image/jpeg"),
                    "reference": (ref_file.name,    ref_bytes,    "image/jpeg"),
                }
                url  = f"{API_BASE}/verify?model={selected_model}"
                resp = requests.post(url, files=files, timeout=60)

                if resp.status_code != 200:
                    st.error(f"API error: {resp.json().get('detail', resp.text)}")
                else:
                    data = resp.json()

                    roi_pil = Image.open(io.BytesIO(base64.b64decode(data["roi_image_b64"]))).convert("L")
                    ref_pil = Image.open(io.BytesIO(base64.b64decode(data["ref_image_b64"]))).convert("L")

                    st.session_state.result = {
                        "label":      data["verdict"],
                        "distance":   data["distance"],
                        "roi":        roi_pil,
                        "ref":        ref_pil,
                        "session_id": data["session_id"],
                    }
                    st.rerun()

            except Exception as e:
                st.session_state.result = None
                st.error(str(e))

    # ── Feedback section ──────────────────────────────────────────────────────────
    if st.session_state.result is not None:
        st.divider()
        st.subheader("🧠 Provide Feedback")

        predicted_label = st.session_state.result["label"]

        corrected_label = st.radio(
            "Correct the prediction if needed:",
            ["GENUINE", "FORGED"],
            index=0 if predicted_label == "GENUINE" else 1,
        )

        if st.button("💾 Save Feedback"):
            if corrected_label != predicted_label:
                fb_resp = requests.post(
                    f"{API_BASE}/feedback",
                    json={
                        "session_id":    st.session_state.result["session_id"],
                        "correct_label": corrected_label,
                        "writer_id":     st.session_state.writer_id,
                    },
                )
                if fb_resp.status_code == 404:
                    st.error("Session expired. Please verify again before submitting feedback.")
                elif fb_resp.status_code != 200:
                    st.error(f"Feedback error: {fb_resp.json().get('detail', fb_resp.text)}")
                else:
                    fb_data = fb_resp.json()
                    if fb_data.get("retraining_triggered"):
                        st.success(f"Feedback saved! 🔄 Retraining triggered with {fb_data['feedback_count']} samples.")
                    else:
                        st.success("Feedback saved successfully!")
                        st.caption(f"Feedback count: {fb_data['feedback_count']}/{MIN_FEEDBACK_SAMPLES} samples needed for retraining.")
            else:
                st.info("Prediction is already correct. No feedback stored.")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE (same as original app4.py — reads local JSON files)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "📊 Dashboard":
    st.title("📊 Model Performance Dashboard")

    metrics_dir = PROJECT_ROOT / "src" / "models" / "metrics"

    if not metrics_dir.exists():
        st.warning("No metrics found yet.")
        st.stop()

    files = list(metrics_dir.glob("*.json"))
    if not files:
        st.warning("No trained models available.")
        st.stop()

    data = []
    for file in files:
        with open(file, "r") as f:
            m = json.load(f)
            data.append({
                "Model":     m.get("model_name", "unknown"),
                "Timestamp": m.get("timestamp", None),
                "Val Loss":  m.get("val",  {}).get("loss",      np.nan),
                "Test Loss": m.get("test", {}).get("loss",      np.nan),
                "Accuracy":  m.get("test", {}).get("accuracy",  np.nan),
                "Precision": m.get("test", {}).get("precision", np.nan),
                "Recall":    m.get("test", {}).get("recall",    np.nan),
                "F1 Score":  m.get("test", {}).get("f1",        np.nan),
            })

    df = pd.DataFrame(data)

    sort_by   = st.selectbox("Sort by", ["F1 Score", "Accuracy", "Precision", "Recall", "Val Loss"])
    ascending = sort_by in ["Val Loss", "Test Loss"]
    df        = df.sort_values(by=sort_by, ascending=ascending)

    st.subheader("📋 Model Comparison")
    st.dataframe(df, use_container_width=True)

    st.success(f"🏆 Best Model: {df.iloc[0]['Model']}")

    st.subheader("📈 Performance Comparison")
    st.bar_chart(df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]])

    st.subheader("🔍 Select Model to Inspect")
    selected_model_dashboard = st.selectbox("Choose model", df["Model"])
    if selected_model_dashboard not in available_models:
        st.warning("⚠️ Model file not found for this metric entry")
    st.json(df[df["Model"] == selected_model_dashboard].iloc[0].to_dict())