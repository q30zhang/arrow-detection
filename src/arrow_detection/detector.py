"""Rule-based arrow detection leveraging contour geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ArrowDetection:
    """Container representing a single detection result."""

    bounding_box: Tuple[int, int, int, int]
    direction: str
    confidence: float

    def as_dict(self) -> dict:
        """Return a JSON-serialisable representation of the detection."""

        x, y, w, h = self.bounding_box
        return {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "direction": self.direction,
            "confidence": float(self.confidence),
        }


class ArrowDetector:
    """Detect square-ish arrows with a pronounced tip using OpenCV primitives."""

    def __init__(
        self,
        min_area: float = 1400.0,
        max_area: float = 7200.0,
        aspect_ratio_range: Tuple[float, float] = (0.65, 1.55),
        canny_thresholds: Tuple[int, int] = (40, 140),
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.canny_thresholds = canny_thresholds

    def detect(self, image: np.ndarray) -> List[ArrowDetection]:
        """Detect arrows in ``image``.

        Parameters
        ----------
        image:
            A ``numpy.ndarray`` in BGR format (as returned by :func:`cv2.imread`).

        Returns
        -------
        list of :class:`ArrowDetection`
            A list of detections sorted from left to right.
        """

        processed = self._preprocess(image)
        contours = self._find_candidate_contours(processed)
        detections = [self._contour_to_detection(image, contour) for contour in contours]
        detections = [d for d in detections if d is not None]
        detections.sort(key=lambda det: det.bounding_box[0])
        return detections[:4]

    def annotate(self, image: np.ndarray, detections: Sequence[ArrowDetection]) -> np.ndarray:
        """Return a copy of ``image`` with detection annotations drawn."""

        output = image.copy()
        for detection in detections:
            x, y, w, h = detection.bounding_box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{detection.direction}:{detection.confidence:.2f}"
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return output

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, *self.canny_thresholds)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        filled = cv2.dilate(closed, kernel, iterations=1)
        return filled

    def _find_candidate_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered: List[np.ndarray] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            filtered.append(contour)
        return filtered

    def _contour_to_detection(self, image: np.ndarray, contour: np.ndarray) -> ArrowDetection | None:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        points = contour.reshape(-1, 2)
        fx, fy = max(points, key=lambda pt: (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
        dx = fx - cx
        dy = fy - cy

        direction: str
        if abs(dx) >= abs(dy):
            direction = "d" if dx > 0 else "a"
        else:
            direction = "s" if dy > 0 else "w"

        dominance = max(abs(dx), abs(dy))
        orthogonal = max(1e-5, min(abs(dx), abs(dy)))
        confidence = float(min(1.0, dominance / orthogonal))

        x, y, w, h = cv2.boundingRect(contour)
        return ArrowDetection((int(x), int(y), int(w), int(h)), direction, confidence)


def detect_from_path(image_path: str) -> List[ArrowDetection]:
    """Convenience function for quick experiments from an image path."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    detector = ArrowDetector()
    return detector.detect(image)
