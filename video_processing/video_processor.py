from typing import List, Set
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

from video_processing.utils import (
    initiate_polygon_zones,
    overlay_typography_from_height,
    TRACK_COLOR_PALETTE,
    ZONE_IN_COLOR_PALETTE,
    ZONE_IN_CONGESTED_OVERLAY_COLOR,
    ZONE_OUT_COLOR_PALETTE,
    ZONE_IN_POLYGONS,
    ZONE_LEG_LABELS,
    ZONE_OUT_POLYGONS,
)
from video_processing.detections_manager import DetectionsManager


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        results_csv_path: str = "results.csv",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        congestion_vehicle_threshold: int = 5,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.congestion_vehicle_threshold = max(1, int(congestion_vehicle_threshold))
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.results_csv_path = results_csv_path
        self.frame_counter = 0
        self.speed_update_frames = 10

        self.model = YOLO(source_weights_path)
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.tracker = sv.ByteTrack(frame_rate=self.video_info.fps)

        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS)
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS)

        text_scale, text_thickness, text_padding, box_thickness = overlay_typography_from_height(
            self.video_info.height
        )
        count_text_scale = float(np.clip(text_scale * 1.05, 0.5, 1.0))

        self.box_annotator = sv.BoxAnnotator(
            color=TRACK_COLOR_PALETTE,
            color_lookup=sv.ColorLookup.TRACK,
            thickness=box_thickness,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=TRACK_COLOR_PALETTE,
            color_lookup=sv.ColorLookup.TRACK,
            text_color=sv.Color.WHITE,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            text_position=sv.Position.TOP_CENTER,
            text_offset=(0, -2),
            border_radius=4,
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=TRACK_COLOR_PALETTE,
            color_lookup=sv.ColorLookup.TRACK,
            position=sv.Position.CENTER,
            trace_length=100,
            thickness=max(2, box_thickness - 1),
        )
        self._count_text_scale = count_text_scale
        self._count_text_thickness = max(1, text_thickness)
        self._label_text_padding = text_padding
        self.detections_manager = DetectionsManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        try:
            if self.target_video_path:
                with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                    for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                        annotated_frame = self.process_frame(frame)
                        sink.write_frame(annotated_frame)
            else:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    cv2.imshow("Processed Video", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if not self.target_video_path:
                cv2.destroyAllWindows()
            self.detections_manager.write_results_csv(
                self.results_csv_path,
                self.video_info.fps,
                ZONE_LEG_LABELS,
            )

    @staticmethod
    def _count_unique_trackers_in_in_zones(
        detections_in_zones: List[sv.Detections],
    ) -> int:
        """Unique tracked vehicles inside any IN zone this frame (avoids double-count if zones overlap)."""
        ids: Set[int] = set()
        for d in detections_in_zones:
            if d.tracker_id is None:
                continue
            for tid in d.tracker_id:
                ids.add(int(tid))
        return len(ids)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        concurrent_in_zone_vehicle_count: int,
        is_congested: bool,
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        frame_rate = self.video_info.fps
        scale = 0.05  # Define the scale based on your video and real-world measurement

        # Initialize the labels list
        labels = []

        for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            if self.frame_counter % self.speed_update_frames == 1:
                speed = self.detections_manager.calculate_speed(tracker_id, (x_center, y_center), frame_rate, scale)
            elif self.frame_counter == 2:
                speed = 10*self.detections_manager.calculate_speed(tracker_id, (x_center, y_center), frame_rate, scale)
            elif tracker_id in self.detections_manager.speeds.keys():
                speed = self.detections_manager.speeds[tracker_id]
            else:
                speed = 0
            speed_disp = max(0.0, float(speed))
            labels.append(f"ID {int(tracker_id)} | {speed_disp:.0f} km/h")

        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels=labels
        )

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            zone_in_draw_color = (
                ZONE_IN_CONGESTED_OVERLAY_COLOR
                if is_congested
                else ZONE_IN_COLOR_PALETTE.colors[i % len(ZONE_IN_COLOR_PALETTE.colors)]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame,
                zone_in.polygon,
                zone_in_draw_color,
                thickness=2,
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame,
                zone_out.polygon,
                ZONE_OUT_COLOR_PALETTE.colors[i % len(ZONE_OUT_COLOR_PALETTE.colors)],
                thickness=2,
            )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        text_color=sv.Color.WHITE,
                        text_scale=self._count_text_scale,
                        text_thickness=self._count_text_thickness,
                        text_padding=max(8, self._label_text_padding),
                        background_color=(
                            ZONE_IN_CONGESTED_OVERLAY_COLOR
                            if is_congested
                            else ZONE_IN_COLOR_PALETTE.colors[
                                zone_in_id % len(ZONE_IN_COLOR_PALETTE.colors)
                            ]
                        ),
                    )

        hud = (
            f"Total IN: {self.detections_manager.total_in()} | "
            f"Total OUT: {self.detections_manager.total_out()}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        hud_scale = float(np.clip(self._count_text_scale * 0.85, 0.45, 1.1))
        hud_thickness = max(1, self._count_text_thickness)
        (tw, th), baseline = cv2.getTextSize(hud, font, hud_scale, hud_thickness)
        pad = max(8, self._label_text_padding)
        x0, y0 = pad, pad + th
        for ox, oy in ((2, 2), (-2, -2), (2, -2), (-2, 2)):
            cv2.putText(
                annotated_frame,
                hud,
                (x0 + ox, y0 + oy),
                font,
                hud_scale,
                (0, 0, 0),
                hud_thickness + 1,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated_frame,
            hud,
            (x0, y0),
            font,
            hud_scale,
            (255, 255, 255),
            hud_thickness,
            cv2.LINE_AA,
        )

        congestion_line = (
            f"CONGESTION DETECTED | In-zones now: {concurrent_in_zone_vehicle_count} "
            f"(>{self.congestion_vehicle_threshold}) | Congestion: HIGH"
            if is_congested
            else (
                f"In-zones now: {concurrent_in_zone_vehicle_count} | "
                f"Congestion: LOW (≤{self.congestion_vehicle_threshold})"
            )
        )
        cong_scale = float(np.clip(hud_scale * 1.05, 0.5, 1.25))
        cong_thickness = max(1, hud_thickness)
        (cw, ch), _ = cv2.getTextSize(congestion_line, font, cong_scale, cong_thickness)
        cx0 = max(pad, (annotated_frame.shape[1] - cw) // 2)
        cy0 = pad + th + ch + pad + ch
        text_bgr = (0, 80, 255) if is_congested else (200, 200, 200)
        for ox, oy in ((2, 2), (-2, -2), (2, -2), (-2, 2)):
            cv2.putText(
                annotated_frame,
                congestion_line,
                (cx0 + ox, cy0 + oy),
                font,
                cong_scale,
                (0, 0, 0),
                cong_thickness + 1,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated_frame,
            congestion_line,
            (cx0, cy0),
            font,
            cong_scale,
            text_bgr,
            cong_thickness,
            cv2.LINE_AA,
        )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_counter += 1
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        concurrent_in = self._count_unique_trackers_in_in_zones(detections_in_zones)
        is_congested = concurrent_in > self.congestion_vehicle_threshold

        detections = self.detections_manager.update(
            detections,
            detections_in_zones,
            detections_out_zones,
            self.frame_counter,
        )
        annotated_frame = self.annotate_frame(
            frame,
            detections,
            concurrent_in_zone_vehicle_count=concurrent_in,
            is_congested=is_congested,
        )
        if self.frame_counter % self.speed_update_frames == 1:
            self.detections_manager.update_positions(detections)
        return annotated_frame