import numpy as np
import supervision as sv
from typing import Iterable, List, Tuple

# Distinct, high-saturation hues for tracks (readable on asphalt / mixed lighting).
TRACK_COLOR_PALETTE = sv.ColorPalette.from_hex(
    [
        "#00D4FF",
        "#FFD967",
        "#FF4D6D",
        "#9B5DE5",
        "#00F5A0",
        "#FE9000",
        "#3A86FF",
        "#FB5607",
        "#06D6A0",
        "#FF006E",
    ]
)

# Entry vs exit zones: cool (approach) vs warm (departure), common in traffic analytics UIs.
ZONE_IN_COLOR_PALETTE = sv.ColorPalette.from_hex(
    ["#1ABC9C", "#27AE60", "#16A085", "#229954"]
)
ZONE_OUT_COLOR_PALETTE = sv.ColorPalette.from_hex(
    ["#E74C3C", "#D35400", "#E67E22", "#C0392B"]
)

# IN zones switch to this when concurrent occupancy exceeds the congestion threshold.
ZONE_IN_CONGESTED_OVERLAY_COLOR = sv.Color.from_hex("#E53935")


def overlay_typography_from_height(video_height: int) -> Tuple[float, int, int, int]:
    """Font scale, text thickness, label padding, and box line thickness from frame height (1080p baseline)."""
    h = max(int(video_height), 360)
    text_scale = float(np.clip(h / 1080.0 * 0.62, 0.44, 0.92))
    text_thickness = max(1, min(2, int(round(text_scale))))
    text_padding = int(np.clip(10 * h / 1080.0, 6, 14))
    box_thickness = max(2, min(4, int(round(h / 360))))
    return text_scale, text_thickness, text_padding, box_thickness

ZONE_IN_POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
]

# Human-readable names for CSV export (index aligns with ZONE_IN_POLYGONS / ZONE_OUT_POLYGONS).
ZONE_LEG_LABELS = ["North", "South", "West", "East"]

def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = (sv.Position.CENTER,),
) -> List[sv.PolygonZone]:
    """Build polygon zones for the current Supervision API (PolygonZone no longer takes frame_resolution_wh)."""
    anchors = tuple(triggering_anchors)
    return [
        sv.PolygonZone(polygon=polygon.astype(np.int64), triggering_anchors=anchors)
        for polygon in polygons
    ]