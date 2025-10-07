# Arrow detection reference solution

This repository contains a light-weight, rule-based detector that localises four
square-ish arrows in each training image and predicts their directions using
only classical computer vision operations. The detector is purposely
interpretable and relies on the dataset's structural constraints:

* Exactly four arrows appear per image and share a similar vertical band.
* Arrows have a near-square footprint (roughly 40–70 px in each dimension).
* Each arrow features a prominent triangular tip, which allows us to infer
  orientation from geometry alone.

## Project layout

```
arrow-detection/
├── src/arrow_detection/        # Python package with the detector implementation
├── scripts/                    # CLI entry points for inference and evaluation
└── training-proc/              # Provided dataset of processed training images
```

## Installation

The code depends only on NumPy and OpenCV. Install them into your preferred
virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python-headless
```

## Running the detector on a single image

```bash
python scripts/detect_arrows.py training-proc/08190218_processed.png \
  --output outputs/08190218_annotated.png \
  --json outputs/08190218_detections.json
```

The script prints a JSON payload of the detections (bounding boxes, predicted
orientation, and a simple confidence heuristic). When `--output` is provided the
image is saved with visual annotations.

## Evaluating against the provided labels

```bash
python scripts/evaluate_dataset.py --limit 25
```

The optional `--limit` flag is helpful for quick sanity checks. Remove it to run
on the full dataset. Evaluation reports the proportion of images whose predicted
four-character direction string matches the label exactly and enumerates the
first few mismatches for inspection.

## How it works

1. **Pre-processing** – Images are blurred slightly before applying a Canny edge
   detector followed by morphological closing/dilation to obtain clean arrow
   blobs.
2. **Candidate selection** – Contours are filtered by area and aspect ratio so
   that only near-square shapes remain.
3. **Direction estimation** – For each candidate contour we compute its centroid
   and locate the farthest contour point, which reliably corresponds to the
   arrow tip. Comparing the tip vector's horizontal and vertical components
   reveals the arrow orientation.
4. **Post-processing** – Detections are sorted left-to-right and truncated to the
   expected four arrows per image.

This deterministic baseline is intentionally simple but establishes a reference
point for more advanced learning-based approaches.
