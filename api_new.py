"""
api.py — FastAPI entry point for Cheque Signature Verification.

Endpoints:
  GET  /health     → system status + current model + feedback count
  GET  /models     → list available verification model checkpoints
  POST /verify     → run verification, return result + session_id
  POST /feedback   → submit correct label, auto-trigger retrain at threshold
"""

from __future__ import annotations

import base64
import io
import sys
import uuid
import time
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.siamese_network import SiameseNetwork
from src.utils.config_loader import training_cfg, retraining_cfg
from src.utils.image_utils import resize_with_padding
from src.preprocessing.background_removal import remove_background

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR    = PROJECT_ROOT / "src" / "models" / "checkpoints"
MODEL_PATH   = MODEL_DIR / "model_margin_1.0.pt"
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
FEEDBACK_CSV = FEEDBACK_DIR / "feedback_pairs.csv"
IMAGES_DIR   = FEEDBACK_DIR / "images"

THRESHOLD   = 0.8
TARGET_SIZE = 256
MIN_SAMPLES = retraining_cfg.training.df_length  # 30

# Only model_margin_*.pt files are valid verification models.
# Others (detector_yolo, roi_model, verifier_siamese) are pipeline models
# and must not be used for signature verification.
def _get_valid_models() -> set[str]:
    return {p.name for p in MODEL_DIR.glob("model_margin*.pt")}


# ── Lazy singletons ────────────────────────────────────────────────────────────
_pipeline: Optional[PreprocessingPipeline] = None
_default_model: Optional[SiameseNetwork]   = None


def get_pipeline() -> PreprocessingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PreprocessingPipeline()
    return _pipeline


def get_default_model() -> SiameseNetwork:
    """Returns the latest verification model checkpoint."""
    global _default_model
    if _default_model is None:
        _default_model = _load_model(_get_latest_checkpoint())
    return _default_model


def _get_latest_checkpoint() -> Path:
    checkpoints = sorted(
        MODEL_DIR.glob("model_margin*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        raise RuntimeError("No verification model checkpoints found in " + str(MODEL_DIR))
    return checkpoints[-1]


def _load_model(path: Path) -> SiameseNetwork:
    m = SiameseNetwork(embedding_size=training_cfg.model.embedding_size)
    state = torch.load(str(path), map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def _reload_default_model():
    """Called after retraining to reload the newest checkpoint as default."""
    global _default_model
    _default_model = _load_model(_get_latest_checkpoint())


# ── Image helpers ──────────────────────────────────────────────────────────────

def _extract_roi(image_bytes: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    result = get_pipeline().run(tmp_path)
    if not result.success or result.roi is None:
        raise RuntimeError(result.error or "Pipeline failed to extract ROI")
    roi = result.roi
    if roi.ndim == 3:
        roi = roi[:, :, 0]
    return roi


def _clean_reference(image_bytes: bytes) -> np.ndarray:
    img_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("L"), dtype=np.uint8)
    cleaned = remove_background(img_np)
    if cleaned.dtype != np.uint8:
        cleaned = (cleaned * 255).clip(0, 255).astype(np.uint8)
    return cleaned


def _to_tensor(img_np: np.ndarray) -> torch.Tensor:
    roi = resize_with_padding(img_np, target_size=TARGET_SIZE)
    roi = roi.astype(np.float32) / 255.0
    return torch.from_numpy(roi).unsqueeze(0).unsqueeze(0)


def _np_to_b64(img_np: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_np.astype(np.uint8), mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _compute_distance(model: SiameseNetwork, roi_np: np.ndarray, ref_np: np.ndarray) -> float:
    with torch.no_grad():
        e1, e2 = model(_to_tensor(roi_np), _to_tensor(ref_np))
        return float(torch.norm(e1 - e2, p=2, dim=1).cpu().item())


# ── Feedback helpers ───────────────────────────────────────────────────────────

def _get_feedback_count() -> int:
    if not FEEDBACK_CSV.exists():
        return 0
    try:
        return len(pd.read_csv(FEEDBACK_CSV))
    except Exception:
        return 0


def _save_feedback_to_csv(
    session_id: str,
    roi_np: np.ndarray,
    ref_np: np.ndarray,
    correct_label: str,
    writer_id: str = "api",
):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    roi_filename = f"{session_id}_roi.png"
    ref_filename = f"{session_id}_ref.png"

    roi_resized = resize_with_padding(roi_np, target_size=TARGET_SIZE)
    ref_resized = resize_with_padding(ref_np, target_size=TARGET_SIZE)

    Image.fromarray(roi_resized.astype(np.uint8), mode="L").save(str(IMAGES_DIR / roi_filename))
    Image.fromarray(ref_resized.astype(np.uint8), mode="L").save(str(IMAGES_DIR / ref_filename))

    new_row = {
        "writer_id": writer_id,
        "image1": f"feedback/images/{roi_filename}",
        "image2": f"feedback/images/{ref_filename}",
        "label": 0 if correct_label == "GENUINE" else 1,
    }

    if FEEDBACK_CSV.exists():
        try:
            df = pd.read_csv(FEEDBACK_CSV)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame([new_row])
    else:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([new_row])

    df.to_csv(FEEDBACK_CSV, index=False)


def _trigger_retraining_background():
    def run():
        try:
            from src.training.retrain import retrain
            print("[API] Retraining started...")
            retrain(
                model_path=MODEL_PATH,
                feedback_dir=FEEDBACK_DIR,
            )
            print("[API] Retraining completed.")

            if FEEDBACK_CSV.exists():
                timestamp = int(time.time())
                archive_path = FEEDBACK_CSV.parent / f"feedback_used_{timestamp}.csv"
                FEEDBACK_CSV.rename(archive_path)
                print(f"[API] Feedback archived: {archive_path.name}")

            _reload_default_model()

        except Exception as e:
            print(f"[API] Retraining failed: {e}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()


# ── In-memory session store ────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cheque Signature Verification API",
    description="Verify cheque signatures and collect feedback for continuous retraining.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class VerifyResponse(BaseModel):
    session_id: str
    verdict: str
    distance: float
    threshold: float
    model_used: str
    feedback_count: int
    message: str
    roi_image_b64: str
    ref_image_b64: str


class FeedbackRequest(BaseModel):
    session_id: str
    correct_label: str
    writer_id: Optional[str] = "api"


class FeedbackResponse(BaseModel):
    message: str
    feedback_count: int
    retraining_triggered: bool


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if the API is running and see current model and feedback status."""
    latest = _get_latest_checkpoint()
    return {
        "status": "ok",
        "model": latest.name,
        "feedback_count": _get_feedback_count(),
        "retrain_threshold": MIN_SAMPLES,
    }


@app.get("/models")
def list_models():
    """
    List all available verification model checkpoints.
    Use the filename as the ?model= parameter in /verify.
    Note: pipeline models (yolo, roi_model, verifier_siamese) are excluded —
    only model_margin_*.pt files are valid for signature verification.
    """
    checkpoints = sorted(
        MODEL_DIR.glob("model_margin*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "models": [p.name for p in checkpoints],
        "latest": checkpoints[0].name if checkpoints else None,
        "total": len(checkpoints),
    }


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    cheque:    UploadFile = File(..., description="Scanned cheque image"),
    reference: UploadFile = File(..., description="Reference signature image"),
    model:     str = Query(
                   default=None,
                   description="Model filename to use e.g. model_margin_1.0.pt — leave empty to use latest."
               ),
):
    """
    Run signature verification.

    - Upload a scanned cheque image and a reference signature image.
    - Optionally pass ?model=filename to choose a specific checkpoint.
      Use GET /models to see available options.
    - Returns verdict (GENUINE/FORGED), distance score, and a session_id.
    - Pass the session_id to POST /feedback to submit a correction.
    """
    try:
        # Model selection with validation
        if model:
            valid = _get_valid_models()
            if model not in valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{model}' is not a valid verification model. "
                           f"Valid options: {sorted(valid)}. "
                           f"Pipeline models (yolo, roi, verifier_siamese) cannot be used here.",
                )
            active_model = _load_model(MODEL_DIR / model)
            model_name   = model
        else:
            active_model = get_default_model()
            model_name   = _get_latest_checkpoint().name

        cheque_bytes = await cheque.read()
        ref_bytes    = await reference.read()

        roi_np   = _extract_roi(cheque_bytes, cheque.filename)
        ref_np   = _clean_reference(ref_bytes)
        distance = _compute_distance(active_model, roi_np, ref_np)
        verdict  = "GENUINE" if distance < THRESHOLD else "FORGED"

        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "roi_np":     roi_np,
            "ref_np":     ref_np,
            "verdict":    verdict,
            "model_used": model_name,
        }

        return VerifyResponse(
            session_id=session_id,
            verdict=verdict,
            distance=round(distance, 4),
            threshold=THRESHOLD,
            model_used=model_name,
            feedback_count=_get_feedback_count(),
            message=f"Distance {distance:.4f} vs threshold {THRESHOLD}.",
            roi_image_b64=_np_to_b64(roi_np),
            ref_image_b64=_np_to_b64(ref_np),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest):
    """
    Submit the correct label for a previous verification.

    - session_id: from the /verify response
    - correct_label: "GENUINE" or "FORGED"
    - writer_id: account number of the cheque owner — used for proper
      train/val/test splitting during retraining. Different cheques from
      different account holders should have different writer_id values.

    Automatically triggers retraining in the background when feedback
    count reaches the threshold (default: 30 samples).
    """
    if body.correct_label not in ("GENUINE", "FORGED"):
        raise HTTPException(
            status_code=400,
            detail="correct_label must be 'GENUINE' or 'FORGED'",
        )

    session = _sessions.get(body.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{body.session_id}' not found. Call /verify first.",
        )

    _save_feedback_to_csv(
        session_id=body.session_id,
        roi_np=session["roi_np"],
        ref_np=session["ref_np"],
        correct_label=body.correct_label,
        writer_id=body.writer_id or "api",
    )

    del _sessions[body.session_id]

    count = _get_feedback_count()
    retrain_triggered = False

    if count >= MIN_SAMPLES:
        _trigger_retraining_background()
        retrain_triggered = True
        print(f"[API] Retraining triggered with {count} samples.")
    else:
        print(f"[API] Feedback saved. {count}/{MIN_SAMPLES} samples collected.")

    return FeedbackResponse(
        message="Feedback recorded." + (" Retraining triggered." if retrain_triggered else ""),
        feedback_count=count,
        retraining_triggered=retrain_triggered,
    )