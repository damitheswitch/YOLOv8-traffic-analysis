# Traffic analysis with YOLOv8 and ByteTrack

This project analyzes traffic at a four-way intersection using **Ultralytics YOLOv8** for detection and **Supervision**’s **ByteTrack** integration for multi-object tracking. Processing lives under `video_processing/`.

## What it does

1. **Vehicle detection and tracking** — Bounding boxes colored per track; short motion **traces** behind each vehicle.
2. **Entry / exit zones** — Four **IN** polygons (approach) and four **OUT** polygons (departure), labeled North / South / West / East in `video_processing/utils.py`. Zone colors reflect entry side; counts show how many vehicles from each entry took each exit.
3. **Speed labels** — Estimated speed (km/h) is updated on a fixed frame interval and shown on each track (pixel-based scale; tune `scale` in `video_processing/video_processor.py` for your scene).
4. **Congestion signal** — When the number of **unique** tracked vehicles inside any IN zone exceeds `--congestion_vehicle_threshold`, the UI marks **HIGH** congestion and adjusts IN-zone styling.
5. **HUD** — On-screen totals for vehicles seen entering (**Total IN**) and those that completed at least one IN→OUT path (**Total OUT**).
6. **CSV export** — After a run, writes per completed crossing: timestamp, tracker id, entry side, exit side, estimated speed (`--results_csv_path`, default `results.csv`).

## Repository layout

| Path | Role |
|------|------|
| `main.py` | CLI entrypoint |
| `video_processing/video_processor.py` | YOLO inference, tracking, annotation, video sink or preview |
| `video_processing/detections_manager.py` | Zone logic, speeds, trip completion, CSV writer |
| `video_processing/utils.py` | Zone polygons, colors, zone labels |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Bash script: creates `data/` and downloads sample video + weights via `gdown` |

Intersection polygons in `utils.py` are hard-coded for the sample resolution; if you use another video, adjust those coordinates.

## Requirements

- Python 3.10+ recommended (project tested with 3.12 on Windows).
- Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include `ultralytics`, `supervision>=0.24.0`, `tqdm`, `gdown`, and `inference` (as listed in `requirements.txt`).

## Sample data (weights + video)

**Linux / macOS / Git Bash on Windows:**

```bash
chmod +x setup.sh
./setup.sh
```

**Windows (PowerShell)** — run the same downloads manually if you do not use Bash:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
gdown -O "data/traffic_analysis.mov" "https://drive.google.com/uc?id=1qadBd7lgpediafCpL_yedGjQPk-FLK-W"
gdown -O "data/traffic_analysis.pt" "https://drive.google.com/uc?id=1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z"
```

Place any other `.pt` / video paths you prefer; the CLI only needs valid file paths.

## How to run

**Write an annotated video** (recommended for long clips):

```bash
python main.py ^
  --source_weights_path data/traffic_analysis.pt ^
  --source_video_path data/traffic_analysis.mov ^
  --target_video_path data/traffic_analysis_result.mov ^
  --results_csv_path results.csv ^
  --confidence_threshold 0.3 ^
  --iou_threshold 0.7 ^
  --congestion_vehicle_threshold 5
```

On bash, use line continuation with `\` instead of `^`.

**Preview in a window** (no output file): omit `--target_video_path`. Press `q` to stop; the CSV is still written at the end.

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--source_weights_path` | *(required)* | Path to YOLO weights (`.pt`) |
| `--source_video_path` | *(required)* | Input video path |
| `--target_video_path` | `None` | Output video; if omitted, OpenCV preview is used |
| `--results_csv_path` | `results.csv` | Output CSV for completed IN→OUT trips |
| `--confidence_threshold` | `0.3` | Detection confidence |
| `--iou_threshold` | `0.7` | NMS IoU threshold |
| `--congestion_vehicle_threshold` | `5` | Congestion when unique vehicles in IN zones is **strictly greater** than this value |

## Credits / upstream

Original demo and ideas are preserved from the upstream traffic-analysis workflow; this tree uses a current **Supervision** `PolygonZone` API and adds congestion reporting, HUD totals, and CSV export.
