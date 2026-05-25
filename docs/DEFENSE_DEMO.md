# Thesis defense demo script (~5 minutes)

Use this order during the presentation. Rehearse once on the **same laptop** you will use in the room.

## Before the session

1. Install dependencies: `pip install -r requirements.txt`
2. Either download sample data (`./setup.sh` or see `data/README.md`) **or** skip YOLO and use the bundled sample CSV only.
3. Optional backup: pre-record an annotated video and keep `output/*.png` open in a folder.

## Path A — Full pipeline (recommended if GPU/time allows)

From the repository root:

```bash
python main.py ^
  --source_weights_path data/traffic_analysis.pt ^
  --source_video_path data/traffic_analysis.mov ^
  --target_video_path data/traffic_analysis_result.mov ^
  --results_csv_path results.csv ^
  --confidence_threshold 0.3 ^
  --congestion_vehicle_threshold 5
```

On bash, replace `^` with `\`.

**Say while it runs:** YOLOv8 detects vehicles, ByteTrack maintains IDs, polygon zones record entry/exit sides, HUD shows totals and congestion.

Then:

```bash
python analyze.py --csv results.csv --out-dir output
```

**Show:** `output/speed_distribution.png`, `traffic_volume_timeline.png`, `entry_exit_flow_heatmap.png`, and `output/summary.txt`.

## Path B — Analysis only (fast fallback, ~30 seconds)

If YOLO is slow or dependencies fail on demo day:

```bash
python analyze.py --csv sample/results.csv --out-dir output
```

**Say:** The live detector produces the same CSV schema; this sample file is from a completed run on the intersection video.

## What to point at on screen

| Artifact | Talking point |
|----------|----------------|
| Annotated video | Tracks, speed labels, zone overlays, congestion HIGH/LOW |
| CSV columns | `timestamp`, `tracker_id`, `entry_side`, `exit_side`, `estimated_speed` |
| Heatmap | Origin–destination matrix (which approach feeds which departure) |
| Speed histogram | Distribution of estimated speeds across completed crossings |
| Volume timeline | Adaptive time bins from crossing timestamps |

## Manual validation checklist

- [ ] `main.py --help` and `analyze.py --help` run without error
- [ ] Sample analysis: `python analyze.py --csv sample/results.csv --out-dir output`
- [ ] Four PNGs + `summary.txt` appear under `output/`
- [ ] Full run (optional): CSV row count matches completed crossings shown in HUD **Total OUT**
- [ ] Limitations ready: see `docs/LIMITATIONS.md`

## If something breaks live

1. Switch to **Path B** (sample CSV).
2. Play a **pre-recorded** annotated video clip.
3. Walk through architecture from `README.md` (mermaid diagram) and thesis slides.
