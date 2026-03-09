from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import torch

# ------------------------------------------------------------
# Make sure we can import src/
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Repo imports (YOUR project code)
# ------------------------------------------------------------
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.siamese_network import SiameseNetwork
from src.utils.config_loader import training_cfg
from src.utils.image_utils import resize_with_padding

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Cheque Signature Verification",
    page_icon="🧾",
    layout="wide",
)

# ------------------------------------------------------------
# Model / threshold config
# ------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "checkpoints" / "model_margin_0.5.pt"
THRESHOLD = 0.8
TARGET_SIZE = 256  # matches SiamesePairDataset

# ------------------------------------------------------------
# Cache pipeline + verifier
# ------------------------------------------------------------

@st.cache_resource
def get_cheque_pipeline():
    return PreprocessingPipeline()

@st.cache_resource
def get_reference_pipeline():
    return PreprocessingPipeline(use_reference=True)

@st.cache_resource
def load_verifier():
    model = SiameseNetwork(
        embedding_size=training_cfg.model.embedding_size
    )

    state_dict = torch.load(str(MODEL_PATH), map_location="cpu")
    # model.cnn.load_state_dict(state_dict["cnn"],strict=True)
    # model.cnn.load_state_dict(state_dict["fc"],strict=True)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

# ------------------------------------------------------------
# YOUR training-style model input preparation
# Based on SiamesePairDataset._get_image()
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Scanned cheque -> YOUR preprocessing pipeline -> ROI
# ------------------------------------------------------------
def extract_signature_roi_from_cheque(uploaded_cheque) -> Image.Image:
    """
    Uses ONLY your preprocessing pipeline on the scanned cheque image.
    """
    pipeline = get_cheque_pipeline()

    suffix = Path(uploaded_cheque.name).suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg"]:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_cheque.getvalue())
        tmp_path = tmp.name

    # YOUR preprocessing pipeline call
    result = pipeline.run(tmp_path)

    if not result.success or result.roi is None:
        raise RuntimeError(
            result.error or "Preprocessing pipeline failed to extract signature ROI."
        )

    roi_np = result.roi

    # Ensure grayscale
    if roi_np.ndim == 3:
        roi_np = roi_np[:, :, 0]

    roi_pil = Image.fromarray(roi_np.astype(np.uint8), mode="L")
    return roi_pil


def preprocess_reference_signature(uploaded_ref) -> Image.Image:
    """
    Uses the reference-signature preprocessing pipeline
    based on reference_signature_preprocessing.yaml.
    """
    pipeline = get_reference_pipeline()

    suffix = Path(uploaded_ref.name).suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg"]:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_ref.getvalue())
        tmp_path = tmp.name

    result = pipeline.run(tmp_path)

    if not result.success or result.roi is None:
        raise RuntimeError(
            result.error or "Reference signature preprocessing failed."
        )

    ref_np = result.roi

    if ref_np.ndim == 3:
        ref_np = ref_np[:, :, 0]

    ref_pil = Image.fromarray(ref_np.astype(np.uint8), mode="L")
    return ref_pil
# ------------------------------------------------------------
# Verification logic
# Matches YOUR evaluate_result.py exactly
# ------------------------------------------------------------
def compute_distance(model, sig1_tensor: torch.Tensor, sig2_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        emb1, emb2 = model(sig1_tensor, sig2_tensor)
        dist = torch.norm(emb1 - emb2, p=2, dim=1)
        return float(dist.cpu().item())

def verify_signature(roi_sig_pil: Image.Image, ref_sig_pil: Image.Image):
    model = load_verifier()

    cheque_np = np.array(roi_sig_pil.convert("L"), dtype=np.uint8)
    ref_np = np.array(ref_sig_pil.convert("L"), dtype=np.uint8)

    cheque_sig_tensor = roi_to_model_tensor(cheque_np)
    ref_sig_tensor = roi_to_model_tensor(ref_np)

    distance = compute_distance(model, cheque_sig_tensor, ref_sig_tensor)
    label = "GENUINE" if distance < THRESHOLD else "FORGED"

    return distance, label

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🧾 Cheque Signature Verification")
st.caption(
    "The scanned cheque image is processed using the cheque preprocessing pipeline to extract the signature. "
    "The reference signature is also processed using a separate reference-signature preprocessing pipeline before verification."
)

st.divider()

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("📤 Upload Inputs")

    with st.container(border=True):
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
            disabled=not (cheque_file and ref_file),
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
    st.caption(f"Model used: {MODEL_PATH.name}")

# ------------------------------------------------------------
# Verify click handler
# ------------------------------------------------------------
if verify_btn:
    with st.spinner("Running preprocessing pipeline and verifying signature..."):
        try:
            # 1) Scanned cheque -> YOUR preprocessing pipeline -> extracted signature ROI
            roi_pil = extract_signature_roi_from_cheque(cheque_file)

            # 2) Reference signature -> direct image load only
            # 2) Reference signature -> reference preprocessing pipeline
            ref_pil = preprocess_reference_signature(ref_file)

            # 3) Verification
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