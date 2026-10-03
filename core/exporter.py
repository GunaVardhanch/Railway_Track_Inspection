from typing import List, Dict, Any
import pandas as pd
import io


CRITICAL_DEFECTS = {
    "Rail Crack",
    "Rail Joint Damage",
    "Rail Misalignment",
    "Broken Sleeper",
    "Missing Sleeper",
    "Missing Fastener"
}


def compute_severity(defect_type: str, confidence: float) -> str:
    """Classify defect severity based on safety impact and confidence."""
    # Normalize extra whitespace so 'Missing  Sleeper' matches 'Missing Sleeper'
    cleaned = " ".join(str(defect_type).strip().split())
    if cleaned in CRITICAL_DEFECTS:
        return "HIGH (CRITICAL)" if confidence >= 0.50 else "MEDIUM"
    return "MEDIUM" if confidence >= 0.60 else "LOW"


def incidents_to_dataframe(incidents: List[Dict[str, Any]], min_persistence: int = 1) -> pd.DataFrame:
    """
    Transforms incident logs into a structured pandas DataFrame with verification status.
    """
    if not incidents:
        return pd.DataFrame(columns=[
            "Track ID",
            "Defect Category",
            "Severity",
            "Status",
            "Peak Confidence",
            "Timestamp (s)",
            "Frames Tracked",
            "Start Frame",
            "End Frame",
            "Bounding Box"
        ])

    rows = []
    for inc in incidents:
        conf = inc.get("max_confidence", 0.0)
        defect = inc.get("defect_type", "Unknown")
        bbox = inc.get("best_bbox", [])
        bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]" if len(bbox) == 4 else "N/A"
        duration = inc.get("duration_frames", 1)
        is_verified = inc.get("is_verified", duration >= min_persistence)
        status_label = "🟢 Verified" if is_verified else f"🟡 Track ({duration}f)"
        
        rows.append({
            "Track ID": inc.get("track_id", "-"),
            "Defect Category": defect,
            "Severity": compute_severity(defect, conf),
            "Status": status_label,
            "Peak Confidence": f"{conf * 100:.1f}%",
            "Timestamp (s)": f"{inc.get('timestamp_sec', 0.0):.2f}s",
            "Frames Tracked": duration,
            "Start Frame": inc.get("start_frame", 0),
            "End Frame": inc.get("end_frame", 0),
            "Bounding Box": bbox_str
        })

    return pd.DataFrame(rows)



def stills_detections_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts still image detections list to a formatted DataFrame.
    """
    if not detections:
        return pd.DataFrame(columns=[
            "Index",
            "Defect Category",
            "Severity",
            "Confidence",
            "Bounding Box (X1, Y1, X2, Y2)",
            "Width x Height (px)"
        ])

    rows = []
    for i, det in enumerate(detections, start=1):
        conf = det.get("confidence", 0.0)
        defect = det.get("class_name", "Unknown")
        bbox = det.get("bbox", [0, 0, 0, 0])
        w = max(0, bbox[2] - bbox[0])
        h = max(0, bbox[3] - bbox[1])

        rows.append({
            "Index": i,
            "Defect Category": defect,
            "Severity": compute_severity(defect, conf),
            "Confidence": f"{conf * 100:.1f}%",
            "Bounding Box (X1, Y1, X2, Y2)": f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})",
            "Width x Height (px)": f"{w} x {h}"
        })

    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Encodes DataFrame to UTF-8 CSV bytes ready for Streamlit st.download_button.
    """
    return df.to_csv(index=False).encode("utf-8")
