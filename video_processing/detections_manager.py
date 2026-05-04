import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import supervision as sv


def _union_all_tracker_ids(counts: Dict[int, Dict[int, Set[int]]]) -> Set[int]:
    out: Set[int] = set()
    for by_in in counts.values():
        for id_set in by_in.values():
            out.update(id_set)
    return out


@dataclass(frozen=True)
class CompletedTrip:
    tracker_id: int
    zone_in_id: int
    zone_out_id: int
    first_seen_frame: int


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}
        self.previous_positions: Dict[int, Tuple[float, float]] = {}  # Tracker ID to (x, y)
        self.speeds: Dict[int, float] = {}  # Tracker ID to speed
        self.tracker_first_seen_frame: Dict[int, int] = {}
        self.completed_trips: List[CompletedTrip] = []

    def total_in(self) -> int:
        """Unique vehicles that have entered any IN zone (may not have reached OUT yet)."""
        return len(self.tracker_id_to_zone_id)

    def total_out(self) -> int:
        """Unique vehicles that completed at least one IN→OUT crossing (union across paths)."""
        return len(_union_all_tracker_ids(self.counts))

    def update_positions(self, detections: sv.Detections):
        for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            self.previous_positions[tracker_id] = (x_center, y_center)

    def calculate_speed(self, tracker_id, new_position, frame_rate, scale):
        if tracker_id not in self.previous_positions:
            return 0
        old_position = self.previous_positions[tracker_id]
        distance_pixels = np.sqrt((new_position[0] - old_position[0])**2 + (new_position[1] - old_position[1])**2)
        distance_real = distance_pixels * scale  # Convert pixel distance to real-world distance
        self.speeds[tracker_id] = distance_real * frame_rate
        return self.speeds[tracker_id] 

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
        frame_index: int,
    ) -> sv.Detections:
        if detections_all.tracker_id is not None:
            for tracker_id in detections_all.tracker_id:
                tid = int(tracker_id)
                self.tracker_first_seen_frame.setdefault(tid, frame_index)

        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                tid = int(tracker_id)
                self.tracker_id_to_zone_id.setdefault(tid, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                tid = int(tracker_id)
                if tid not in self.tracker_id_to_zone_id:
                    continue
                zone_in_id = self.tracker_id_to_zone_id[tid]
                self.counts.setdefault(zone_out_id, {})
                self.counts[zone_out_id].setdefault(zone_in_id, set())
                path_ids = self.counts[zone_out_id][zone_in_id]
                if tid not in path_ids:
                    path_ids.add(tid)
                    first_frame = self.tracker_first_seen_frame.get(tid, frame_index)
                    self.completed_trips.append(
                        CompletedTrip(
                            tracker_id=tid,
                            zone_in_id=zone_in_id,
                            zone_out_id=zone_out_id,
                            first_seen_frame=first_frame,
                        )
                    )

        detections_all.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(int(x), -1)
        )(detections_all.tracker_id)
        return detections_all[detections_all.class_id != -1]

    def write_results_csv(
        self,
        path: str | Path,
        fps: float,
        zone_labels: Optional[Sequence[str]] = None,
    ) -> None:
        """Write one row per completed IN→OUT crossing using latest speed estimates."""
        path = Path(path)
        labels = list(zone_labels) if zone_labels is not None else []

        def side_label(zone_id: int) -> str:
            if 0 <= zone_id < len(labels):
                return labels[zone_id]
            return str(zone_id)

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp",
                    "tracker_id",
                    "entry_side",
                    "exit_side",
                    "estimated_speed",
                ]
            )
            for trip in self.completed_trips:
                ts = (trip.first_seen_frame - 1) / fps if fps > 0 else 0.0
                speed = float(self.speeds.get(trip.tracker_id, 0.0))
                w.writerow(
                    [
                        f"{ts:.4f}",
                        trip.tracker_id,
                        side_label(trip.zone_in_id),
                        side_label(trip.zone_out_id),
                        f"{speed:.2f}",
                    ]
                )

