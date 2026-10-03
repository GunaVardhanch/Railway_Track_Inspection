import gc
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import torch
from ultralytics import RTDETR

from config import (
    DEFAULT_WEIGHTS,
    DEFAULT_CONF,
    DEFAULT_LINE_THICKNESS,
    DEFAULT_FONT_SCALE
)


# Distinct color palette for railway defects (RGB) inspired by RailSamiksha severity system
CLASS_COLORS = {
    0: (6, 182, 212),     # Ballast Deficiency - Cyber Cyan
    1: (251, 146, 60),    # Ballast Washout - Amber Orange
    2: (244, 63, 94),     # Broken Sleeper - Crimson Rose (Critical)
    3: (234, 179, 8),     # Damaged Fastener - Electric Gold
    4: (236, 72, 153),    # Missing Sleeper - Magenta (Critical)
    5: (245, 158, 11),    # Missing Fastener - Vivid Amber
    6: (239, 68, 68),     # Rail Crack - Laser Red (Critical)
    7: (249, 115, 22),    # Rail Joint Damage - Solar Orange (High)
    8: (59, 130, 246),    # Rail Misalignment - Aerospace Blue
}


def draw_defect_annotation(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    line_thickness: int = DEFAULT_LINE_THICKNESS,
    font_scale: float = DEFAULT_FONT_SCALE,
    auto_scale: bool = True
) -> None:
    """
    Renders an enhanced, high-visibility bounding box and high-contrast label pill
    for defect inspection outputs.
    
    Args:
        image: Image array (in-place modification).
        box: (x1, y1, x2, y2) coordinates.
        label: Text string to display (e.g. defect name and confidence).
        color: Color tuple in the same color order as image (BGR or RGB).
        line_thickness: Base line thickness in pixels.
        font_scale: Base font scale for cv2.putText.
        auto_scale: If True, dynamically scales thickness and font proportionally to image resolution.
    """
    h, w = image.shape[:2]
    
    if auto_scale:
        scale_factor = max(1.0, min(w, h) / 720.0)
        actual_thickness = max(2, int(line_thickness * scale_factor))
        actual_font_scale = font_scale * max(1.0, scale_factor * 0.85)
    else:
        actual_thickness = max(1, int(line_thickness))
        actual_font_scale = float(font_scale)
        
    actual_font_thickness = max(1, int(actual_font_scale * 2.2))

    x1, y1, x2, y2 = [int(v) for v in box]
    
    # Draw primary bounding box with prominent thickness
    cv2.rectangle(image, (x1, y1), (x2, y2), color, actual_thickness)
    
    # Calculate text dimensions
    (t_w, t_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, actual_font_thickness)
    
    pad_x = max(6, int(8 * (actual_font_scale / 0.85)))
    pad_y = max(5, int(6 * (actual_font_scale / 0.85)))
    pill_w = t_w + 2 * pad_x
    pill_h = t_h + baseline + 2 * pad_y
    
    # Compute high-contrast text color based on background color luminance
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    text_color = (0, 0, 0) if luminance > 155 else (255, 255, 255)
    
    # Position label pill above box if space permits, otherwise place inside top of box
    if y1 - pill_h >= 0:
        p1 = (x1, y1 - pill_h)
        p2 = (min(w, x1 + pill_w), y1)
        text_org = (x1 + pad_x, y1 - pad_y - baseline // 2)
    else:
        p1 = (x1, y1)
        p2 = (min(w, x1 + pill_w), min(h, y1 + pill_h))
        text_org = (x1 + pad_x, y1 + pad_y + t_h)
        
    cv2.rectangle(image, p1, p2, color, -1)
    cv2.putText(
        image,
        label,
        text_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        actual_font_scale,
        text_color,
        actual_font_thickness,
        cv2.LINE_AA
    )


class RTDETREngine:
    """
    Ultralytics RT-DETR model wrapper supporting direct native inference
    and ByteTrack persistent tracking for railway defect inspection.
    """
    def __init__(self, weights_path: str = DEFAULT_WEIGHTS, default_conf: float = DEFAULT_CONF):
        self.weights_path = weights_path
        self.conf = default_conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Restrict PyTorch thread workspace memory on CPU to prevent OpenMP heap exhaustion
        torch.set_num_threads(1)
        
        # Load RT-DETR model onto selected device
        self.model = RTDETR(self.weights_path)
        if self.device == "cuda":
            self.model.to("cuda")
            
        self.names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

    def infer_direct(
        self,
        frame_bgr: np.ndarray,
        imgsz: int = 640,
        conf: Optional[float] = None,
        line_thickness: int = DEFAULT_LINE_THICKNESS,
        font_scale: float = DEFAULT_FONT_SCALE,
        auto_scale: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Single-pass native RT-DETR inference on standard input frame (BGR).
        Returns:
            annotated_rgb: np.ndarray (RGB format ready for Streamlit display)
            detections: List of dicts with keys (class_id, class_name, confidence, bbox, track_id)
        """
        effective_conf = conf if conf is not None else self.conf
        detections: List[Dict[str, Any]] = []
        annotated_bgr = frame_bgr.copy()

        with torch.inference_mode():
            results = self.model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=effective_conf,
                device=self.device,
                verbose=False
            )
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, score, cls_id in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cls_name = self.names.get(cls_id, f"Class {cls_id}")
                    color = CLASS_COLORS.get(cls_id % len(CLASS_COLORS), (0, 255, 0))
                    # OpenCV uses BGR
                    color_bgr = (color[2], color[1], color[0])

                    label = f"{cls_name} {score:.2f}"
                    draw_defect_annotation(
                        annotated_bgr,
                        (x1, y1, x2, y2),
                        label=label,
                        color=color_bgr,
                        line_thickness=line_thickness,
                        font_scale=font_scale,
                        auto_scale=auto_scale
                    )

                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": cls_name,
                        "confidence": float(score),
                        "bbox": [x1, y1, x2, y2],
                        "track_id": None
                    })

            del results, result

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        del annotated_bgr
        return annotated_rgb, detections
