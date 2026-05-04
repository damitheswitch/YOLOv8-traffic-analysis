import argparse
from video_processing.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--results_csv_path",
        default="results.csv",
        help="Path for per-vehicle CSV (timestamp, tracker_id, entry/exit side, speed)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    parser.add_argument(
        "--congestion_vehicle_threshold",
        default=5,
        type=int,
        help="Declare congestion when unique vehicles in all IN zones exceed this count (strictly greater than N)",
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        results_csv_path=args.results_csv_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        congestion_vehicle_threshold=args.congestion_vehicle_threshold,
    )
    processor.process_video()

if __name__ == "__main__":
    main()
