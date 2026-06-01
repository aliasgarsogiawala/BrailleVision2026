from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


def _cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    configured = [origin.strip() for origin in raw.split(",") if origin.strip()]
    defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    ordered: List[str] = []
    for origin in [*configured, *defaults]:
        if origin not in ordered:
            ordered.append(origin)
    return ordered


app = FastAPI(title="BrailleVision API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ROI_SCALE = 0.70
ROW_BUCKET_TOLERANCE = 15.0
MIN_CONTOUR_AREA = 12.0
MAX_CONTOUR_AREA = 80.0
MIN_SOLIDITY = 0.90


GRADE_1_MAP: Dict[Tuple[int, int, int, int, int, int], str] = {
    (1, 0, 0, 0, 0, 0): "a",
    (1, 1, 0, 0, 0, 0): "b",
    (1, 0, 0, 1, 0, 0): "c",
    (1, 0, 0, 1, 1, 0): "d",
    (1, 0, 0, 0, 1, 0): "e",
    (1, 1, 0, 1, 0, 0): "f",
    (1, 1, 0, 1, 1, 0): "g",
    (1, 1, 0, 0, 1, 0): "h",
    (0, 1, 0, 1, 0, 0): "i",
    (0, 1, 0, 1, 1, 0): "j",
    (1, 0, 1, 0, 0, 0): "k",
    (1, 1, 1, 0, 0, 0): "l",
    (1, 0, 1, 1, 0, 0): "m",
    (1, 0, 1, 1, 1, 0): "n",
    (1, 0, 1, 0, 1, 0): "o",
    (1, 1, 1, 1, 0, 0): "p",
    (1, 1, 1, 1, 1, 0): "q",
    (1, 1, 1, 0, 1, 0): "r",
    (0, 1, 1, 1, 0, 0): "s",
    (0, 1, 1, 1, 1, 0): "t",
    (1, 0, 1, 0, 0, 1): "u",
    (1, 1, 1, 0, 0, 1): "v",
    (0, 1, 0, 1, 1, 1): "w",
    (1, 0, 1, 1, 0, 1): "x",
    (1, 0, 1, 1, 1, 1): "y",
    (1, 0, 1, 0, 1, 1): "z",
    (0, 0, 0, 0, 0, 0): " ",
}


@dataclass
class DotCandidate:
    x: int
    y: int
    w: int
    h: int
    area: float
    solidity: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def diameter(self) -> float:
        return float((self.w + self.h) / 2.0)


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    matrix = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(matrix, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Unable to decode image frame.")
    return frame


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def _center_roi_bounds(width: int, height: int) -> Tuple[int, int, int, int]:
    roi_width = int(width * ROI_SCALE)
    roi_height = int(height * ROI_SCALE)
    left = max(0, (width - roi_width) // 2)
    top = max(0, (height - roi_height) // 2)
    right = min(width, left + roi_width)
    bottom = min(height, top + roi_height)
    return left, top, right, bottom


def _encode_debug_image(mask: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        return ""
    return f"data:image/png;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def _extract_dot_candidates(mask: np.ndarray) -> List[DotCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[DotCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0.0:
            continue

        solidity = area / hull_area
        if solidity <= MIN_SOLIDITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        candidates.append(
            DotCandidate(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=area,
                solidity=solidity,
            )
        )

    candidates.sort(key=lambda dot: (dot.center[1], dot.center[0]))
    return candidates


def _bucket_rows(candidates: List[DotCandidate]) -> List[List[DotCandidate]]:
    if not candidates:
        return []

    rows: List[List[DotCandidate]] = []
    for candidate in candidates:
        cy = candidate.center[1]
        if not rows:
            rows.append([candidate])
            continue

        placed = False
        for row in rows:
            reference_y = float(np.median([dot.center[1] for dot in row]))
            if abs(cy - reference_y) <= ROW_BUCKET_TOLERANCE:
                row.append(candidate)
                placed = True
                break

        if not placed:
            rows.append([candidate])

    filtered = [sorted(row, key=lambda dot: dot.center[0]) for row in rows if len(row) >= 2]
    filtered.sort(key=lambda row: float(np.median([dot.center[1] for dot in row])))
    return filtered


def _estimate_dot_scale(rows: List[List[DotCandidate]]) -> float:
    diameters = [dot.diameter for row in rows for dot in row]
    if not diameters:
        return 8.0
    return float(np.median(diameters))


def _estimate_horizontal_step(row: List[DotCandidate], diameter: float) -> float:
    centers_x = [dot.center[0] for dot in row]
    gaps = [centers_x[index + 1] - centers_x[index] for index in range(len(centers_x) - 1)]
    valid = [gap for gap in gaps if gap > diameter * 0.6]
    if not valid:
        return max(diameter * 2.5, 18.0)
    return float(np.median(valid))


def _hamming_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(int(left != right) for left, right in zip(a, b))


def _match_pattern(pattern: Sequence[int]) -> Tuple[Tuple[int, int, int, int, int, int], str, int]:
    pattern_tuple = tuple(int(bit) for bit in pattern)
    if pattern_tuple in GRADE_1_MAP:
        return pattern_tuple, GRADE_1_MAP[pattern_tuple], 0

    best = min(GRADE_1_MAP.keys(), key=lambda candidate: _hamming_distance(pattern_tuple, candidate))
    return best, GRADE_1_MAP[best], _hamming_distance(pattern_tuple, best)


def _expected_points(cell_left: float, row_y: float, step: float) -> List[Tuple[float, float]]:
    return [
        (cell_left, row_y),
        (cell_left, row_y + step),
        (cell_left, row_y + 2.0 * step),
        (cell_left + step, row_y),
        (cell_left + step, row_y + step),
        (cell_left + step, row_y + 2.0 * step),
    ]


def _cell_pattern(
    cell_left: float,
    row_y: float,
    dots: List[DotCandidate],
    step: float,
) -> Tuple[Tuple[int, int, int, int, int, int], str, int, List[int]]:
    points = _expected_points(cell_left, row_y, step)
    radius = max(step * 0.85, 10.0)
    pattern = [0, 0, 0, 0, 0, 0]

    for index, (px, py) in enumerate(points):
        for dot in dots:
            dx, dy = dot.center
            if float(np.hypot(dx - px, dy - py)) <= radius:
                pattern[index] = 1
                break

    matched_pattern, translated, distance = _match_pattern(pattern)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    box = [
        max(0, int(round(min(xs) - radius))),
        max(0, int(round(min(ys) - radius))),
        max(1, int(round((max(xs) - min(xs)) + (2.0 * radius)))),
        max(1, int(round((max(ys) - min(ys)) + (2.0 * radius)))),
    ]
    return matched_pattern, translated, distance, box


def _decode_rows(rows: List[List[DotCandidate]]) -> Tuple[str, float, List[List[int]]]:
    if not rows:
        return "", 0.0, []

    diameter = _estimate_dot_scale(rows)
    text_lines: List[str] = []
    confidences: List[float] = []
    boxes: List[List[int]] = []

    for row in rows:
        row_y = float(np.median([dot.center[1] for dot in row]))
        step = max(_estimate_horizontal_step(row, diameter), diameter * 2.3)
        starts = [row[0].center[0]]

        for index in range(1, len(row)):
            prev_x = row[index - 1].center[0]
            curr_x = row[index].center[0]
            gap = curr_x - prev_x
            if gap > step * 1.6:
                estimated_slots = max(1, int(round(gap / step)) - 1)
                for slot in range(estimated_slots):
                    starts.append(prev_x + step * (slot + 1))
            starts.append(curr_x)

        merged_starts: List[float] = []
        for start in starts:
            if not merged_starts or abs(start - merged_starts[-1]) > step * 0.55:
                merged_starts.append(start)

        row_text = ""
        previous_start: float | None = None
        for start in merged_starts:
            if previous_start is not None and start - previous_start > step * 1.8:
                row_text += " "

            matched_pattern, translated, distance, box = _cell_pattern(start, row_y, row, step)
            row_text += translated
            boxes.append(box)

            occupancy = sum(matched_pattern) / 6.0
            closeness = max(0.45, 1.0 - (distance / 6.0))
            confidence = min(1.0, 0.18 + occupancy * 0.42 + closeness * 0.4)
            confidences.append(confidence)
            previous_start = start

        cleaned = row_text.strip()
        if cleaned:
            text_lines.append(cleaned)

    text = "\n".join(text_lines).strip()
    confidence = round(float(np.mean(confidences)), 3) if confidences else 0.0
    return text, confidence, boxes


def process_braille_frame(frame: np.ndarray) -> Tuple[str, float, List[List[int]], str]:
    normalized = _resize_frame(frame)
    frame_height, frame_width = normalized.shape[:2]
    roi_left, roi_top, roi_right, roi_bottom = _center_roi_bounds(frame_width, frame_height)
    roi = normalized[roi_top:roi_bottom, roi_left:roi_right]

    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    equalized = clahe.apply(grayscale)
    thresholded = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        101,
        2,
    )
    denoised = cv2.medianBlur(thresholded, 5)

    candidates = _extract_dot_candidates(denoised)
    rows = _bucket_rows(candidates)
    text, confidence, roi_boxes = _decode_rows(rows)
    mapped_boxes = [[x + roi_left, y + roi_top, w, h] for x, y, w, h in roi_boxes]
    debug_image = _encode_debug_image(denoised)
    return text, confidence, mapped_boxes, debug_image


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/process-frame")
async def process_frame(file: UploadFile = File(...)) -> Dict[str, object]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Received empty frame.")

    frame = _decode_image(raw)
    text, confidence, boxes, debug_image = process_braille_frame(frame)
    return {
        "text": text,
        "confidence": confidence,
        "boxes": boxes,
        "debug_image": debug_image,
    }
