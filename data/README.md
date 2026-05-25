# Sample data

This folder is **not** tracked in git (videos and weights are large). Download the demo assets here before running the pipeline.

## Quick setup

**Linux / macOS / Git Bash:**

```bash
chmod +x setup.sh
./setup.sh
```

**Windows (PowerShell)** — from the repository root:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
gdown -O "data/traffic_analysis.mov" "https://drive.google.com/uc?id=1qadBd7lgpediafCpL_yedGjQPk-FLK-W"
gdown -O "data/traffic_analysis.pt" "https://drive.google.com/uc?id=1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z"
```

## Expected files

| File | Purpose |
|------|---------|
| `traffic_analysis.mov` | Sample four-way intersection video |
| `traffic_analysis.pt` | YOLOv8 weights fine-tuned for this scene |

After download, run `main.py` with paths under `data/` (see root `README.md`).

If you only need thesis figures without running YOLO, use the bundled sample CSV instead:

```bash
python analyze.py --csv sample/results.csv --out-dir output
```
