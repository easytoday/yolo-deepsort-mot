# Object Detection and Tracking in Videos — YOLO + Deep SORT

A multiple object tracking (MOT) system for detecting and tracking pedestrians in
video sequences, based on the *tracking-by-detection* paradigm. The detector
(YOLOv3 or YOLOv8) localizes pedestrians in each frame; Deep SORT assigns them a
persistent identity over time.

Evaluated on the **MOTSChallenge / MOT17** benchmark using the standard MOTA, MOTP,
and IDF1 metrics.

## Demo

| YOLOv3 (baseline) | YOLOv8 (optimized) |
|:---:|:---:|
| ![YOLOv3](docs/yolov3.gif) | ![YOLOv8](docs/yolov8.gif) |
| MOTA 57.0% · IDF1 54.4% | MOTA 62.6% · IDF1 65.0% |

## Architecture

```
Video sequence → YOLOv3/YOLOv8 (detection) → Deep SORT (tracking) → MOT results → evaluation
```

The detector and tracker are decoupled through a stable interface based on bounding
boxes, allowing the detector to be swapped without modifying the rest of the
pipeline.


## Project Structure

All scripts are run from the repository root (e.g. `python scripts/run_tracker.py ...`).

```
yolo_deepsort_mot/
├── environment.yml          # conda environment
├── configs/                 # detector, tracker and experiment configurations
│   ├── yolov3.yaml
│   ├── yolov8.yaml
│   ├── deepsort.yaml
│   └── experiments.yaml     # ablation study definition (12 configurations)
├── src/                     # library code (detection + tracking)
│   ├── detector.py          # YOLOv3 wrapper (OpenCV DNN)
│   ├── detector_yolov8.py   # YOLOv8 wrapper (ultralytics)
│   ├── detector_factory.py  # detector selector
│   ├── tracker.py           # Deep SORT wrapper
│   └── utils.py             # MOT17 reading, result writing
├── scripts/                 # all runnable entry points
│   ├── run_tracker.py       # tracking on a single sequence
│   ├── run_mot17.py         # batch tracking
│   ├── run_ablation.py      # experimental study
│   ├── evaluate.py          # metrics computation
│   ├── generate_figures.py  # report figures (FR + EN)
│   ├── make_video.py        # annotated video generation
│   ├── make_gif.sh          # video → GIF conversion
│   ├── setup.sh             # download external dependencies
│   ├── download_mot17.sh    # dataset download helper
│   ├── convert_mots_to_mot17.py  # MOTS → MOT bounding-box conversion
│   └── verify_install.py    # installation check
├── tests/
│   └── test_pipeline.py     # pipeline smoke tests
├── figures/                 # generated report figures (FR)
├── figures_en/              # generated report figures (EN)
└── docs/                    # demo GIFs
```

The following are **not tracked in the repository** and are created locally on
demand:

```
├── deep_sort/               # external dependency, cloned by scripts/setup.sh
├── weights/                 # model weights, downloaded by scripts/setup.sh
├── data/                    # MOT17 / MOTSChallenge dataset (see Data Preparation)
├── results/                 # experiment outputs (reproducible)
└── videos/                  # generated annotated videos
```

## Installation

```bash
# 1. Clone the repository
git clone <REPOSITORY_URL>
cd yolo_deepsort_mot

# 2. Create the conda environment
conda env create -f environment.yml
conda activate mot

# 3. Download external dependencies (Deep SORT, YOLOv3 weights, ReID model)
bash scripts/setup.sh

# 4. Verify the installation
python scripts/verify_install.py
```

The YOLOv8 detector automatically downloads its weights (`yolov8s.pt`) on first use,
via the `ultralytics` package.

## Data Preparation

The MOTSChallenge benchmark is downloaded from RWTH Aachen University
(motchallenge.net is no longer available).

```bash
# Show download sources
bash scripts/download_mot17.sh --sources

# Download MOTSChallenge.zip then extract it
wget -P data/ https://www.vision.rwth-aachen.de/media/resource_files/MOTSChallenge.zip
bash scripts/download_mot17.sh --file data/MOTSChallenge.zip

# Convert MOTS annotations (RLE segmentation) into MOT bounding boxes
python scripts/convert_mots_to_mot17.py
```

This conversion is required because MOTSChallenge provides segmentation masks,
whereas bounding-box evaluation expects rectangular coordinates. RLE decoding relies
on `pycocotools`.

## Usage

### Tracking on a single sequence

```bash
# With YOLOv3 (default detector)
python scripts/run_tracker.py --sequence data/MOT17/train/MOT17-02-DPM

# With YOLOv8
python scripts/run_tracker.py --sequence data/MOT17/train/MOT17-02-DPM --detector yolov8
```

### Processing all sequences

```bash
python scripts/run_mot17.py --split train --detector yolov8
```

### Evaluation

```bash
python scripts/evaluate.py --split train
```

## Reproducing the Full Experimental Study

The study compares 12 configurations (8 for YOLOv3, 4 for YOLOv8) covering inference
resolution, confidence threshold, tracking parameters, and the choice of detector.
It is driven by `scripts/run_ablation.py`, which records progress and allows the study to be
run incrementally over time.

```bash
# Show progress
python scripts/run_ablation.py --status

# Run a specific experiment
python scripts/run_ablation.py --exp v8_combo

# Run an entire axis (e.g. all YOLOv8 configurations)
python scripts/run_ablation.py --axis detector_v8

# Run all remaining experiments
python scripts/run_ablation.py --all

# Generate the final comparison table
python scripts/run_ablation.py --report
```

Experiments are defined in `configs/experiments.yaml`. Each result is saved under
`results/ablation/<id>/` together with the exact configuration used
(`config_used.yaml`), ensuring reproducibility.

## Results

Evaluation on 4 MOTSChallenge sequences (2862 frames), IoU threshold 0.5.

| Configuration | Detector | MOTA | MOTP | IDF1 |
|:---|:---:|:---:|:---:|:---:|
| Baseline | YOLOv3 | 57.0% | 0.211 | 54.4% |
| Optimized (resolution 608) | YOLOv3 | 60.6% | 0.204 | 57.4% |
| Optimized (combined) | YOLOv3 | 60.3% | 0.206 | 62.0% |
| Native (640) | YOLOv8 | 61.9% | 0.144 | 61.1% |
| **Optimized (combined)** | **YOLOv8** | **62.6%** | **0.152** | **65.0%** |

Key findings: inference resolution is the most accessible lever (though subject to
an optimum), and the detector is the most decisive one. YOLOv8s proved roughly twice
as fast as YOLOv3 on CPU, while improving all quality metrics.

## Generating Demo Videos and GIFs

```bash
# 1. Generate an annotated video from existing results (fast)
python scripts/make_video.py \
    --sequence data/MOT17/train/MOT17-02-DPM \
    --results results/ablation/v8_combo/MOT17-02-DPM.txt \
    --output videos/mot17-02_yolov8.avi

python scripts/make_video.py \
    --sequence data/MOT17/train/MOT17-02-DPM \
    --results results/ablation/baseline/MOT17-02-DPM.txt \
    --output videos/mot17-02_yolov3.avi

# 2. Convert to a lightweight GIF (~3-4 MB, 10-second clip)
bash scripts/make_gif.sh videos/mot17-02_yolov8.avi docs/yolov8.gif
bash scripts/make_gif.sh videos/mot17-02_yolov3.avi docs/yolov3.gif
```

## Technology Stack

- **YOLOv3** — Redmon & Farhadi, 2018 (arXiv:1804.02767)
- **YOLOv8** — Ultralytics, 2023
- **Deep SORT** — Wojke, Bewley & Paulus, 2017 (arXiv:1703.07402)
- **MOTSChallenge** — Voigtlaender et al., 2019 (CVPR)

## License

This project is released under the MIT License.
