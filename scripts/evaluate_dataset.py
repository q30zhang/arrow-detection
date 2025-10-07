#!/usr/bin/env python3
"""Evaluate the rule-based detector against the supplied label file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2

from arrow_detection import ArrowDetector


@dataclass
class EvaluationResult:
    image_path: Path
    expected: str
    predicted: str
    detections: List[dict]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("training-proc"),
        help="Directory containing processed training PNG files",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("training-proc/labels.json"),
        help="Path to the JSON file containing labels",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of images to evaluate (useful for smoke tests)",
    )
    return parser


def _iter_labelled_images(images_dir: Path, label_path: Path) -> Iterable[EvaluationResult]:
    data = json.loads(label_path.read_text())
    detector = ArrowDetector()
    for idx, entry in enumerate(data):
        prefix = entry["image_prefix"]
        expected = entry["labels"]
        image_path = images_dir / f"{prefix}_processed.png"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read {image_path}")
        detections = detector.detect(image)
        predicted = "".join(det.direction for det in detections)
        yield EvaluationResult(
            image_path=image_path,
            expected=expected,
            predicted=predicted,
            detections=[det.as_dict() for det in detections],
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    results = []
    for idx, result in enumerate(_iter_labelled_images(args.images, args.labels)):
        results.append(result)
        if args.limit is not None and idx + 1 >= args.limit:
            break

    total = len(results)
    exact_matches = sum(1 for res in results if res.expected == res.predicted)
    print(f"Processed {total} images")
    print(f"Exact matches: {exact_matches}/{total} ({exact_matches / max(1, total):.2%})")

    mismatches = [res for res in results if res.expected != res.predicted]
    if mismatches:
        print("\nFirst 5 mismatches:")
        for res in mismatches[:5]:
            print(f"- {res.image_path.name}: expected {res.expected}, predicted {res.predicted}")


if __name__ == "__main__":
    main()
