"""Inference logic: YOLOv8-Seg crack detection with optional SAHI slicing."""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# torch>=2.6 defaults weights_only=True, which rejects ultralytics checkpoints.
_orig_load = torch.load
torch.load = lambda f, **kw: _orig_load(f, **{"weights_only": False, **kw})

SAHI_MODEL_IMGSZ = 640  # net input size; slices smaller than this get upscaled (magnified)
SAHI_SLICE_SIZE = 512
SAHI_OVERLAP = 0.2
CONF_THRESHOLD = 0.25
WEIGHTS_PATH = Path(__file__).parent / "weights" / "best.pt"


def load_model(weights: str | None = None) -> YOLO:
    """Load YOLOv8-Seg model for standard (full-frame) inference."""
    path = weights or str(WEIGHTS_PATH)
    if not Path(path).exists():
        raise FileNotFoundError(f"Weights not found: {path}. Run: python download_weights.py")
    return YOLO(path)


def _retina_sahi_class() -> Any:
    """SAHI ultralytics model that requests full-resolution (retina) masks.

    Stock SAHI calls the YOLO model without retina_masks, so masks come back at
    the ~160px prototype resolution and smear to fill the bbox on thin diagonal
    cracks. retina_masks=True returns masks at input resolution = tight traces.
    """
    from sahi.models.ultralytics import UltralyticsDetectionModel

    class RetinaUltralyticsModel(UltralyticsDetectionModel):
        def perform_batch_inference(self, images: list[np.ndarray]) -> None:
            if self.model is None:
                raise ValueError("Model is not loaded, load it by calling .load_model()")
            kwargs = {
                "cfg": self.config_path, "verbose": False,
                "conf": self.confidence_threshold, "device": self.device,
                "retina_masks": True,
            }
            if self.image_size is not None:
                kwargs = {"imgsz": self.image_size, **kwargs}
            images_bgr = [img[:, :, ::-1] for img in images]
            prediction_result = self.model(images_bgr, **kwargs)
            self._original_predictions = self._extract_predictions(prediction_result)
            self._original_shapes = [img.shape for img in images]

    return RetinaUltralyticsModel


def load_sahi_model(conf: float = CONF_THRESHOLD, weights: str | None = None) -> Any:
    """Load weights wrapped as a SAHI detection model for sliced inference."""
    path = weights or str(WEIGHTS_PATH)
    if not Path(path).exists():
        raise FileNotFoundError(f"Weights not found: {path}. Run: python download_weights.py")
    return _retina_sahi_class()(
        model_path=path,
        confidence_threshold=conf,
        image_size=SAHI_MODEL_IMGSZ,
    )


def run_inference(
    image: np.ndarray,
    model: Any,
    use_sahi: bool = False,
    conf: float = CONF_THRESHOLD,
    slice_size: int = SAHI_SLICE_SIZE,
) -> tuple[np.ndarray, int]:
    """Run inference and return annotated image + detection count.

    When use_sahi is True, `model` must be a SAHI detection model
    (see load_sahi_model); otherwise it must be a YOLO model.
    `slice_size` only applies to the SAHI path: smaller than the model's
    640 input means each tile is upscaled, magnifying thin cracks.
    """
    if use_sahi:
        return _infer_sahi(image, model, slice_size)
    return _infer_standard(image, model, conf)


def _infer_standard(image: np.ndarray, model: YOLO, conf: float) -> tuple[np.ndarray, int]:
    results = model.predict(image, conf=conf, retina_masks=True, verbose=False)
    annotated = results[0].plot()  # BGR
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated, counts(results)


def _infer_sahi(image: np.ndarray, sahi_model: Any, slice_size: int) -> tuple[np.ndarray, int]:
    """Sliced inference via the SAHI library: tile, predict, merge with NMS."""
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import visualize_object_predictions

    result = get_sliced_prediction(
        image,
        sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=SAHI_OVERLAP,
        overlap_width_ratio=SAHI_OVERLAP,
        verbose=0,
    )
    preds = result.object_prediction_list
    vis = visualize_object_predictions(
        image.copy(),
        object_prediction_list=preds,
        rect_th=2,
        text_size=0.4,
    )
    return vis["image"], len(preds)


def counts(results) -> int:
    """Count total detections (masks) across all results."""
    if hasattr(results, "masks") and results.masks is not None:
        return len(results.masks)
    if hasattr(results, "__iter__"):
        return sum(
            len(r.masks) for r in results if hasattr(r, "masks") and r.masks is not None
        )
    return 0


def download_weights(hf_token: str | None = None) -> None:
    """Download best.pt from HuggingFace."""
    from huggingface_hub import hf_hub_download

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        "johnnytan5/crack-seg-yolov8s",
        "best.pt" if not hf_token else "runs/full/exp/weights/best.pt",
        local_dir=str(WEIGHTS_PATH.parent),
    )
    if Path(path) != WEIGHTS_PATH:
        import shutil
        shutil.copy(path, WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")
