#!/usr/bin/env python3
"""CLI utility for running the rule-based arrow detector on an image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from arrow_detection import ArrowDetector


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the image to analyse")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path where an annotated image should be written",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path where detections should be serialised as JSON",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image {args.image} does not exist")

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Failed to load image {args.image}")

    detector = ArrowDetector()
    detections = detector.detect(image)

    if args.output:
        annotated = detector.annotate(image, detections)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), annotated)

    serialised: List[dict] = [det.as_dict() for det in detections]
    json_payload = json.dumps(serialised, indent=2)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json_payload)
    else:
        print(json_payload)


if __name__ == "__main__":
    main()
