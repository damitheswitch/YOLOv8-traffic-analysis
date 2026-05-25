# Known limitations

These are intentional scope boundaries for the thesis prototype, not bugs to hide in the defense.

## Scene-specific geometry

Entry and exit zones are **hard-coded polygons** in `video_processing/utils.py`, tuned for the sample video resolution (~1920×1080). A new camera angle or resolution requires redrawing those coordinates.

## Speed estimation

Speed is derived from **pixel displacement** between frames, scaled by a constant (`scale = 0.05` in `video_processing/video_processor.py`). Values are shown as km/h for readability but are **not** radar- or GPS-calibrated. They support relative comparisons (fast vs. slow tracks) and aggregate statistics, not enforcement-grade measurement.

## Congestion signal

Congestion is declared when the count of **unique vehicles inside any IN zone** exceeds `--congestion_vehicle_threshold`. This is a simple occupancy heuristic, not a traffic-engineering model (no queue length, signal timing, or density per lane).

## Model and classes

The bundled weights (`traffic_analysis.pt`) target the demo intersection. Detection quality depends on lighting, occlusion, and vehicle types seen during training or fine-tuning.

## Manual validation

The project is validated by **running the pipeline on the sample video** and inspecting the annotated output, CSV rows, and `analyze.py` figures. There is no continuous integration or automated regression suite (see `docs/DEFENSE_DEMO.md` for the manual check list).
