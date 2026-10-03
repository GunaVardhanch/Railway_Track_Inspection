import gc
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Generator, Tuple, Optional
from config import (
    MIN_PERSISTENCE_FRAMES,
    DEFAULT_CONF,
    DEFAULT_STRIDE,
    DEFAULT_LINE_THICKNESS,
    DEFAULT_FONT_SCALE,
    DEFAULT_VIDEO_IMGSZ
)
from core.detector import RTDETREngine, CLASS_COLORS, draw_defect_annotation


def compute_bbox_iou(box1: List[int], box2: List[int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = float(box1_area + box2_area - inter_area)

    if union <= 0.0:
        return 0.0
    return inter_area / union


class VideoInspectionPipeline:
    """
    Video processing pipeline that executes ByteTrack persistent tracking with RT-DETR,
    maintains an incident registry, and enforces temporal persistence filtering
    to eliminate false alarms with strict CPU/GPU memory safety.
    """
    def __init__(
        self,
        video_path: str,
        detector: RTDETREngine,
        conf: float = DEFAULT_CONF,
        stride: int = DEFAULT_STRIDE,
        min_persistence_frames: int = MIN_PERSISTENCE_FRAMES,
        imgsz: int = DEFAULT_VIDEO_IMGSZ,
        line_thickness: int = DEFAULT_LINE_THICKNESS,
        font_scale: float = DEFAULT_FONT_SCALE,
        auto_scale: bool = True
    ):
        self.video_path = video_path
        self.detector = detector
        self.conf = conf
        self.stride = max(1, stride)
        self.min_persistence = max(1, min_persistence_frames)
        self.imgsz = imgsz
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.auto_scale = auto_scale

        # Incident registry keyed by track_id
        # {track_id: {track_id, class_id, defect_type, max_confidence, start_frame, end_frame, timestamp_sec, duration_frames, best_bbox, is_verified}}
        self.incident_registry: Dict[int, Dict[str, Any]] = {}
        
        # Internal tracker state for spatial matching
        self._recent_tracks: Dict[int, Dict[str, Any]] = {}
        self._next_synthetic_id: int = 100

    def process_stream(self, stop_event: Optional[Any] = None) -> Generator[Tuple[int, int, np.ndarray, List[Dict[str, Any]]], None, None]:
        """
        Streams processed video frames with RT-DETR detection and spatial tracking.
        Args:
            stop_event: Optional threading.Event or object with is_set() method to signal graceful stop.
        Yields:
            (current_frame_idx, total_frames, annotated_rgb_frame, active_detections_list)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video stream at: {self.video_path}")

        raw_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = max(0, int(raw_total)) if (raw_total is not None and not np.isnan(raw_total)) else 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        current_frame_idx = 0
        processed_count = 0

        # Maximum working frame dimension (ultra-low memory footprint for CPU stability)
        MAX_FRAME_DIM = 480

        # Restrict PyTorch thread workspace memory on low-RAM hosts
        torch.set_num_threads(1)

        try:
            while cap.isOpened():
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    break

                current_frame_idx += 1

                # Frame stride sampling: use grab() to advance demuxer without decoding raw frames or allocating memory
                if (current_frame_idx - 1) % self.stride != 0:
                    if not cap.grab():
                        break
                    continue

                try:
                    ret, raw_frame_bgr = cap.read()
                except Exception:
                    gc.collect()
                    continue

                if not ret or raw_frame_bgr is None:
                    break

                processed_count += 1
                timestamp_sec = round(current_frame_idx / fps, 2)

                # Cap resolution to keep memory and CPU usage stable
                raw_h, raw_w = raw_frame_bgr.shape[:2]
                if max(raw_h, raw_w) > MAX_FRAME_DIM:
                    scale = MAX_FRAME_DIM / float(max(raw_h, raw_w))
                    proc_frame = cv2.resize(raw_frame_bgr, (int(raw_w * scale), int(raw_h * scale)), interpolation=cv2.INTER_AREA)
                    del raw_frame_bgr
                else:
                    scale = 1.0
                    proc_frame = raw_frame_bgr

                active_detections: List[Dict[str, Any]] = []
                annotated_bgr = proc_frame.copy()

                # Run native RT-DETR prediction within PyTorch inference mode (zero autograd/tracker overhead)
                with torch.inference_mode():
                    results = self.detector.model.predict(
                        source=proc_frame,
                        conf=self.conf,
                        imgsz=min(480, self.imgsz),
                        device=self.detector.device,
                        verbose=False
                    )

                    result = results[0]

                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes_np = result.boxes.xyxy.cpu().numpy()
                        confs_np = result.boxes.conf.cpu().numpy()
                        classes_np = result.boxes.cls.cpu().numpy().astype(int)
                    else:
                        boxes_np, confs_np, classes_np = [], [], []

                    # Explicitly release Ultralytics PyTorch result buffers
                    del results, result

                # Prune inactive tracks beyond expiry window to keep spatial matching O(active) and bound memory
                track_expiry = self.stride * 12
                self._recent_tracks = {
                    tid: trk for tid, trk in self._recent_tracks.items()
                    if (current_frame_idx - trk["last_frame"]) <= track_expiry
                }
                matched_ids_in_frame: set = set()

                # Process detections
                if len(boxes_np) > 0:
                    for box, score, cls_id in zip(boxes_np, confs_np, classes_np):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        raw_cls_name = self.detector.names.get(cls_id, f"Defect_{cls_id}")
                        cls_name = " ".join(raw_cls_name.strip().split())
                        color = CLASS_COLORS.get(cls_id % len(CLASS_COLORS), (0, 255, 0))
                        color_bgr = (color[2], color[1], color[0])

                        # Resolve persistent Track ID using spatial IoU continuity with collision prevention
                        best_iou = 0.0
                        best_match_id = None
                        for existing_id, trk in self._recent_tracks.items():
                            if existing_id in matched_ids_in_frame:
                                continue
                            if trk["class_id"] == cls_id and (current_frame_idx - trk["last_frame"]) <= (self.stride * 4):
                                iou = compute_bbox_iou([x1, y1, x2, y2], trk["bbox"])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match_id = existing_id
                        
                        if best_match_id is not None and best_iou >= 0.15:
                            t_id = best_match_id
                        else:
                            t_id = self._next_synthetic_id
                            self._next_synthetic_id += 1

                        matched_ids_in_frame.add(t_id)

                        # Compute original-scale bbox for export reporting
                        orig_bbox = [
                            int(x1 / scale),
                            int(y1 / scale),
                            int(x2 / scale),
                            int(y2 / scale)
                        ]

                        # Always register every detection in incident registry
                        if t_id not in self.incident_registry:
                            self.incident_registry[t_id] = {
                                "track_id": t_id,
                                "class_id": int(cls_id),
                                "defect_type": cls_name,
                                "max_confidence": float(score),
                                "start_frame": current_frame_idx,
                                "end_frame": current_frame_idx,
                                "timestamp_sec": timestamp_sec,
                                "duration_frames": 1,
                                "best_bbox": orig_bbox,
                                "is_verified": (1 >= self.min_persistence)
                            }
                        else:
                            rec = self.incident_registry[t_id]
                            rec["duration_frames"] += 1
                            rec["end_frame"] = current_frame_idx
                            if score > rec["max_confidence"]:
                                rec["max_confidence"] = float(score)
                                rec["best_bbox"] = orig_bbox
                            rec["is_verified"] = (rec["duration_frames"] >= self.min_persistence)

                        is_verified = self.incident_registry[t_id]["is_verified"]
                        duration = self.incident_registry[t_id]["duration_frames"]

                        # Cache recent track state for spatial matching
                        self._recent_tracks[t_id] = {
                            "bbox": [x1, y1, x2, y2],
                            "class_id": int(cls_id),
                            "last_frame": current_frame_idx
                        }

                        active_detections.append({
                            "track_id": t_id,
                            "class_id": int(cls_id),
                            "defect_type": cls_name,
                            "confidence": float(score),
                            "bbox": orig_bbox,
                            "duration_frames": duration,
                            "is_verified": is_verified,
                            "timestamp_sec": timestamp_sec
                        })

                        # Visual annotation on frame
                        status_tag = "" if is_verified else f" [{duration}/{self.min_persistence}]"
                        label = f"#{t_id} {cls_name} {score:.2f}{status_tag}"

                        draw_defect_annotation(
                            annotated_bgr,
                            (x1, y1, x2, y2),
                            label=label,
                            color=color_bgr,
                            line_thickness=self.line_thickness,
                            font_scale=self.font_scale,
                            auto_scale=self.auto_scale
                        )

                # Overlay HUD telemetry on video
                verified_count = len(self.get_verified_incidents())
                total_detected = len(self.incident_registry)
                frame_label = f"{current_frame_idx}/{total_frames}" if total_frames > 0 else f"{current_frame_idx}"
                hud_text = f"Frame {frame_label} | Time: {timestamp_sec:.1f}s | In-View: {len(active_detections)} | Total: {total_detected} (Verified: {verified_count})"
                
                # Draw sleek dark aerospace HUD bar bounded safely by frame width
                frame_w = annotated_bgr.shape[1]
                hud_font_scale = 0.44 if frame_w < 500 else 0.48
                (hud_w, hud_h), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, 1)
                hud_box_right = min(frame_w - 6, 20 + hud_w)
                cv2.rectangle(annotated_bgr, (6, 6), (hud_box_right, 34), (7, 11, 18), -1)
                cv2.rectangle(annotated_bgr, (6, 6), (hud_box_right, 34), (30, 41, 59), 1)
                cv2.putText(annotated_bgr, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (56, 189, 248), 1, cv2.LINE_AA)

                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                del annotated_bgr, proc_frame

                yield current_frame_idx, total_frames, annotated_rgb, active_detections

                # Garbage collection on every frame to guarantee contiguous free CPU blocks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            cap.release()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_all_incidents(self, sort_by_recent: bool = True) -> List[Dict[str, Any]]:
        """
        Returns all unique recorded incidents regardless of duration,
        sorted by most recently observed by default.
        """
        incidents = list(self.incident_registry.values())
        if sort_by_recent:
            incidents.sort(key=lambda x: (x["end_frame"], x["start_frame"]), reverse=True)
        else:
            incidents.sort(key=lambda x: x["start_frame"])
        return incidents

    def get_verified_incidents(self, sort_by_recent: bool = False) -> List[Dict[str, Any]]:
        """
        Returns only deduplicated incidents meeting the minimum temporal persistence filter threshold.
        """
        verified = [
            inc for inc in self.incident_registry.values()
            if inc.get("is_verified", False) or inc["duration_frames"] >= self.min_persistence
        ]
        if sort_by_recent:
            verified.sort(key=lambda x: (x["end_frame"], x["start_frame"]), reverse=True)
        else:
            verified.sort(key=lambda x: x["start_frame"])
        return verified
