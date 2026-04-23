from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import time
import numpy as np
import streamlit as st
from PIL import Image
import threading

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_RELOAD_FLAG = threading.Event()

from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.siamese_network import SiameseNetwork
from src.utils.config_loader import training_cfg,retraining_cfg
from src.utils.image_utils import resize_with_padding

def get_available_models():
    models_dir = PROJECT_ROOT / "src" / "models" / "checkpoints"
    return sorted([f.name for f in models_dir.glob("*.pt")])

def get_current_model_path():
    return PROJECT_ROOT / "src" / "models" / "checkpoints" / st.session_state.selected_model

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "model_margin_1.0.pt"

if "result" not in st.session_state:
    st.session_state.result = None

if MODEL_RELOAD_FLAG.is_set():
    st.cache_resource.clear()
    MODEL_RELOAD_FLAG.clear()
    st.rerun()

st.set_page_config(
    page_title="Cheque Signature Verification",
    page_icon="🧾",
    layout="wide",
)

st.sidebar.title("🧠 Model Control")

mode = st.sidebar.radio(
    "Select Mode",
    ["🔍 Inference", "📊 Dashboard"]
)

available_models = get_available_models()

if st.session_state.selected_model not in available_models:
    st.session_state.selected_model = available_models[0]

selected_model = st.sidebar.selectbox(
    "Select Model Version",
    available_models,
    index=available_models.index(st.session_state.selected_model)
)

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.cache_resource.clear()
    st.rerun()

#MODEL_PATH = PROJECT_ROOT / "src" / "models" / "checkpoints" / "model_margin_1.0.pt"
THRESHOLD = 0.8
TARGET_SIZE = 256

@st.cache_resource


def get_pipeline():
    return PreprocessingPipeline()

@st.cache_resource

def load_verifier():
    model = SiameseNetwork(
        embedding_size=training_cfg.model.embedding_size
    )

    model_path = PROJECT_ROOT / "src" / "models" / "checkpoints" / st.session_state.selected_model
    state_dict = torch.load(str(model_path), map_location="cpu")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def clean_reference_signature(pil_img: Image.Image) -> Image.Image:
    """
    For reference signatures cropped from a cheque scan.
    Applies only background removal to clean the image,
    no full pipeline (no ROI extraction, no geometric corrections).
    """
    import cv2
    from src.preprocessing.background_removal import remove_background

    img_np = np.array(pil_img.convert("L"), dtype=np.uint8)
    cleaned = remove_background(img_np)

    if cleaned.dtype != np.uint8:
        cleaned = (cleaned * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(cleaned, mode="L")

def roi_to_model_tensor(roi_np: np.ndarray) -> torch.Tensor:
    """
    Matches your SiamesePairDataset logic:
      roi = resize_with_padding(roi, target_size=256)
      roi = roi.astype(np.float32) / 255.0
      roi = torch.from_numpy(roi).unsqueeze(0)

    For inference we add batch dimension too:
      [H,W] -> [1,1,H,W]
    """
    roi = resize_with_padding(roi_np, target_size=TARGET_SIZE)
    roi = roi.astype(np.float32) / 255.0
    roi = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0)
    return roi

def pil_signature_to_model_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Used for reference signature only.
    No cheque preprocessing pipeline.
    Just convert to grayscale numpy and apply the same final
    prep used in your dataset.
    """
    img = pil_img.convert("L")
    img_np = np.array(img, dtype=np.uint8)
    return roi_to_model_tensor(img_np)

def extract_signature_roi_from_cheque(uploaded_cheque) -> Image.Image:
    """
    Uses ONLY your preprocessing pipeline on the scanned cheque image.
    """
    pipeline = get_pipeline()

    suffix = Path(uploaded_cheque.name).suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg"]:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_cheque.getvalue())
        tmp_path = tmp.name

    result = pipeline.run(tmp_path)

    if not result.success or result.roi is None:
        raise RuntimeError(
            result.error or "Preprocessing pipeline failed to extract signature ROI."
        )

    roi_np = result.roi

    if roi_np.ndim == 3:
        roi_np = roi_np[:, :, 0]

    roi_pil = Image.fromarray(roi_np.astype(np.uint8), mode="L")
    return roi_pil

def compute_distance(model, sig1_tensor: torch.Tensor, sig2_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        emb1, emb2 = model(sig1_tensor, sig2_tensor)
        dist = torch.norm(emb1 - emb2, p=2, dim=1)
        return float(dist.cpu().item())

def verify_signature(roi_sig_pil: Image.Image, ref_sig_pil: Image.Image):
    model = load_verifier()

    cheque_np = np.array(roi_sig_pil.convert("L"), dtype=np.uint8)
    cheque_sig_tensor = roi_to_model_tensor(cheque_np)

    ref_sig_tensor = pil_signature_to_model_tensor(ref_sig_pil)

    distance = compute_distance(model, cheque_sig_tensor, ref_sig_tensor)
    label = "GENUINE" if distance < THRESHOLD else "FORGED"

    return distance, label

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
                help="Used to ensure proper data splitting during retraining"
            )
            st.session_state.writer_id = account_number

            cheque_file = st.file_uploader(
                "Scanned Cheque Image",
                type=["png", "jpg", "jpeg"],
                help="This full cheque image will go through your preprocessing pipeline and signature extraction.",
            )

            ref_file = st.file_uploader(
                "Reference Signature",
                type=["png", "jpg", "jpeg"],
                help="Upload a cropped reference signature image.",
            )

            verify_btn = st.button(
                "✅ Verify Signature",
                use_container_width=True,
                disabled=not (cheque_file and ref_file and account_number),
            )

            if not (cheque_file and ref_file):
                st.info("Upload both the cheque image and the reference signature to enable verification.")

        if cheque_file:
            cheque_preview = Image.open(cheque_file).convert("RGB")
            st.image(cheque_preview, caption="Uploaded Scanned Cheque", use_container_width=True)

        if ref_file:
            ref_preview = Image.open(ref_file).convert("RGB")
            st.image(ref_preview, caption="Uploaded Reference Signature", use_container_width=True)

    with right:
        st.subheader("📌 Result")

        if "result" not in st.session_state:
            st.session_state.result = None

        with st.container(border=True):
            if st.session_state.result is None:
                st.write("Upload both images and click **Verify Signature**.")
                st.caption("The extracted signature from the cheque and final result will appear here.")
            else:
                label = st.session_state.result["label"]
                distance = st.session_state.result["distance"]
                roi = st.session_state.result["roi"]
                ref = st.session_state.result["ref"]

                if label == "GENUINE":
                    st.success("✅ GENUINE")
                else:
                    st.error("⚠️ FORGED")

                st.metric("Euclidean Distance", f"{distance:.4f}")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(roi, caption="Extracted Signature (from Cheque Pipeline)", use_container_width=True)
                with c2:
                    st.image(ref, caption="Reference Signature", use_container_width=True)

        st.caption(f"Decision rule: distance < {THRESHOLD:.2f} → GENUINE, otherwise FORGED.")
        st.caption(f"Model used: {st.session_state.selected_model}")

    if verify_btn:
        with st.spinner("Running preprocessing pipeline and verifying signature..."):
            try:
                roi_pil = extract_signature_roi_from_cheque(cheque_file)

                ref_raw = Image.open(ref_file)
                ref_pil = clean_reference_signature(ref_raw)

                distance, label = verify_signature(roi_pil, ref_pil)

                st.session_state.result = {
                    "label": label,
                    "distance": distance,
                    "roi": roi_pil,
                    "ref": ref_pil,
                }
                st.rerun()

            except Exception as e:
                st.session_state.result = None
                st.error(str(e))


    import uuid
    import threading
    import pandas as pd

    FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
    FEEDBACK_CSV = FEEDBACK_DIR / "feedback_pairs.csv"
    MIN_FEEDBACK_SAMPLES = retraining_cfg.training.df_length

    def get_feedback_count():
        if not FEEDBACK_CSV.exists():
            return 0
        return len(pd.read_csv(FEEDBACK_CSV))


    def should_retrain():
        return get_feedback_count() >= MIN_FEEDBACK_SAMPLES

    def save_feedback(roi_pil, ref_pil, label):
        sample_id = str(uuid.uuid4())

        images_dir = FEEDBACK_DIR / "images"
        images_dir.mkdir(parents=True, exist_ok=True) 

        roi_filename = f"{sample_id}_roi.png"
        ref_filename = f"{sample_id}_ref.png"

        roi_path = images_dir / roi_filename   
        ref_path = images_dir / ref_filename

        roi_rel = f"feedback/images/{roi_filename}"  
        ref_rel = f"feedback/images/{ref_filename}"

        roi_np = resize_with_padding(np.array(roi_pil.convert("L")), target_size=256)
        ref_np = resize_with_padding(np.array(ref_pil.convert("L")), target_size=256)

        Image.fromarray(roi_np.astype(np.uint8), mode="L").save(str(roi_path))
        Image.fromarray(ref_np.astype(np.uint8), mode="L").save(str(ref_path))

        assert roi_path.exists(), f"ROI image was not saved: {roi_path}"
        assert ref_path.exists(), f"Ref image was not saved: {ref_path}"

        print(f"[INFO] Images saved to: {images_dir}")

        new_row = {
            "writer_id": st.session_state.get("writer_id", "unknown"),
            "image1": roi_rel,
            "image2": ref_rel,
            "label": 0 if label == "GENUINE" else 1
        }

        if FEEDBACK_CSV.exists():
            try:
                df = pd.read_csv(FEEDBACK_CSV)

                if df.empty:
                    df = pd.DataFrame([new_row])
                else:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            except pd.errors.EmptyDataError:
                df = pd.DataFrame([new_row])

        else:
            df = pd.DataFrame([new_row])

        df.to_csv(FEEDBACK_CSV, index=False)
        print(f"[INFO] Feedback saved: {FEEDBACK_CSV}")

    def trigger_retraining(model_path: Path):
        def run():
            try:
                from src.training.retrain import retrain
                print("[INFO] Starting retraining...")
                retrain(
                    model_path=model_path,              
                    feedback_dir=PROJECT_ROOT / "data" / "feedback"
                )

                print("[INFO] Retraining completed.")
                feedback_csv = PROJECT_ROOT / "data" / "feedback" / "feedback_pairs.csv"

                if feedback_csv.exists():
                    timestamp = int(time.time())
                    archive_path = feedback_csv.parent / f"feedback_used_{timestamp}.csv"

                    feedback_csv.rename(archive_path)
                    print(f"[INFO] Feedback data archived: {archive_path.name}")

                MODEL_RELOAD_FLAG.set()

            except Exception as e:
                print("[ERROR] Retraining failed:", e)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    if st.session_state.get("result") is not None:
        st.divider()
        st.subheader("🧠 Provide Feedback")

        predicted_label = st.session_state.result["label"]

        corrected_label = st.radio(
            "Correct the prediction if needed:",
            ["GENUINE", "FORGED"],
            index=0 if predicted_label == "GENUINE" else 1
        )

        if st.button("💾 Save Feedback"):
            predicted_label = st.session_state.result["label"]

            if corrected_label != predicted_label:

                save_feedback(
                    st.session_state.result["roi"],
                    st.session_state.result["ref"],
                    corrected_label
                )

                count = get_feedback_count()

                if should_retrain():
                    model_path = get_current_model_path()
                    trigger_retraining(model_path)
                    print(f"[INFO] Retraining triggered with {count} samples.")
                else:
                    print(f"[INFO] Feedback saved: {count}/{MIN_FEEDBACK_SAMPLES}")

                st.success("Feedback saved successfully!")

            else:
                print("[INFO] Feedback matches prediction. Not storing.")
                st.info("Prediction is already correct. No feedback stored.")

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

    import json
    import pandas as pd

    data = []

    for file in files:
        with open(file, "r") as f:
            m = json.load(f)

            row = {
                "Model": m.get("model_name", "unknown"),
                "Timestamp": m.get("timestamp", None),
                "Val Loss": m.get("val", {}).get("loss", np.nan),
                "Test Loss": m.get("test", {}).get("loss", np.nan),
                "Accuracy": m.get("test", {}).get("accuracy", np.nan),
                "Precision": m.get("test", {}).get("precision", np.nan),
                "Recall": m.get("test", {}).get("recall", np.nan),
                "F1 Score": m.get("test", {}).get("f1", np.nan)
            }
            data.append(row)

    df = pd.DataFrame(data)

    # Sorting
    sort_by = st.selectbox(
        "Sort by",
        ["F1 Score", "Accuracy", "Precision", "Recall", "Val Loss"]
    )

    ascending = sort_by in ["Val Loss", "Test Loss"]
    df = df.sort_values(by=sort_by, ascending=ascending)

    st.subheader("📋 Model Comparison")
    st.dataframe(df, use_container_width=True)

    # Best model
    best_model = df.iloc[0]
    st.success(f"🏆 Best Model: {best_model['Model']}")

    # Chart
    st.subheader("📈 Performance Comparison")
    st.bar_chart(
        df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]
    )

    # Model selector
    st.subheader("🔍 Select Model")
    selected_model_dashboard = st.selectbox(
        "Choose model to inspect",
        df["Model"]
    )
    if selected_model_dashboard not in available_models:
        st.warning("⚠️ Model file not found for this metric entry")

    selected_row = df[df["Model"] == selected_model_dashboard].iloc[0]
    st.json(selected_row.to_dict())