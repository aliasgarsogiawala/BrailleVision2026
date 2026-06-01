from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, Body, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


def _cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "")
    configured = [origin.strip() for origin in raw.split(",") if origin.strip()]
    defaults = [
        "http://localhost:3000",
        "https://braille-vision2026-web.vercel.app",
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
ROI_SCALE = 1.0
INNER_PAPER_SCALE = 0.82
ROW_BUCKET_TOLERANCE = 15.0
MIN_CONTOUR_AREA = 18.0
MAX_CONTOUR_AREA = 180.0
MIN_SOLIDITY = 0.90
MIN_CIRCULARITY = 0.75
MIN_CONTRAST = 10.0


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
    circularity: float
    contrast: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def diameter(self) -> float:
        return float((self.w + self.h) / 2.0)


class Base64ImagePayload(BaseModel):
    image: str


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    matrix = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(matrix, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Unable to decode image frame.")
    return frame


def _decode_base64_image(image_data: str) -> np.ndarray:
    if not image_data or not image_data.strip():
        raise HTTPException(status_code=400, detail="No Base64 image data was provided.")

    encoded = image_data.strip()
    if "," in encoded:
        _, encoded = encoded.split(",", 1)

    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid Base64 image payload.") from exc

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Decoded Base64 image was empty.")

    return _decode_image(raw_bytes)


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


def _inner_focus_mask(grayscale: np.ndarray) -> np.ndarray:
    height, width = grayscale.shape[:2]
    focus_width = int(width * INNER_PAPER_SCALE)
    focus_height = int(height * INNER_PAPER_SCALE)
    left = max(0, (width - focus_width) // 2)
    top = max(0, (height - focus_height) // 2)
    right = min(width, left + focus_width)
    bottom = min(height, top + focus_height)

    mask = np.zeros_like(grayscale, dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def _extract_dot_candidates(mask: np.ndarray, grayscale: np.ndarray) -> List[DotCandidate]:
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

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.8 or aspect_ratio > 1.25:
            continue

        contour_mask = np.zeros(grayscale.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        contour_pixels = grayscale[contour_mask == 255]
        if contour_pixels.size == 0:
            continue

        ring_mask = np.zeros(grayscale.shape, dtype=np.uint8)
        cv2.circle(
            ring_mask,
            (int(x + w / 2), int(y + h / 2)),
            max(4, int(max(w, h) * 1.8)),
            255,
            thickness=-1,
        )
        ring_mask = cv2.subtract(ring_mask, contour_mask)
        ring_pixels = grayscale[ring_mask == 255]
        background_mean = float(np.mean(ring_pixels)) if ring_pixels.size else float(np.mean(grayscale))
        foreground_mean = float(np.mean(contour_pixels))
        contrast = max(0.0, background_mean - foreground_mean)
        if contrast < MIN_CONTRAST:
            continue

        candidates.append(
            DotCandidate(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=area,
                solidity=solidity,
                circularity=circularity,
                contrast=contrast,
            )
        )

    candidates.sort(key=lambda dot: (dot.center[1], dot.center[0]))
    return candidates


def _refine_candidates(candidates: List[DotCandidate]) -> List[DotCandidate]:
    if len(candidates) < 2:
        return []

    diameters = np.array([dot.diameter for dot in candidates], dtype=np.float32)
    areas = np.array([dot.area for dot in candidates], dtype=np.float32)
    median_diameter = float(np.median(diameters))
    median_area = float(np.median(areas))

    filtered: List[DotCandidate] = []
    for dot in candidates:
        if dot.diameter < median_diameter * 0.7 or dot.diameter > median_diameter * 1.6:
            continue
        if dot.area < median_area * 0.45 or dot.area > median_area * 2.2:
            continue

        neighbor_count = 0
        cx, cy = dot.center
        for other in candidates:
            if other is dot:
                continue
            ox, oy = other.center
            if abs(ox - cx) <= median_diameter * 3.5 and abs(oy - cy) <= median_diameter * 1.6:
                neighbor_count += 1
        if neighbor_count >= 1:
            filtered.append(dot)

    filtered.sort(key=lambda dot: (dot.center[1], dot.center[0]))
    return filtered


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

    filtered_rows: List[List[DotCandidate]] = []
    for row in rows:
        if len(row) < 2:
            continue

        ordered = sorted(row, key=lambda dot: dot.center[0])
        row_diameter = float(np.median([dot.diameter for dot in ordered]))
        max_gap = max(row_diameter * 4.2, 48.0)

        current_segment: List[DotCandidate] = [ordered[0]]
        for dot in ordered[1:]:
            prev = current_segment[-1]
            if dot.center[0] - prev.center[0] > max_gap:
                if len(current_segment) >= 2:
                    filtered_rows.append(current_segment)
                current_segment = [dot]
            else:
                current_segment.append(dot)

        if len(current_segment) >= 2:
            filtered_rows.append(current_segment)

    filtered_rows.sort(key=lambda row: float(np.median([dot.center[1] for dot in row])))
    return filtered_rows


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
    paper_mask = _inner_focus_mask(grayscale)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    equalized = clahe.apply(grayscale)
    filtered = cv2.bilateralFilter(equalized, 7, 40, 40)
    thresholded = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        91,
        3,
    )

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, blackhat_kernel)
    blackhat = cv2.GaussianBlur(blackhat, (5, 5), 0)
    _, blackhat_mask = cv2.threshold(
        blackhat,
        int(max(10, np.percentile(blackhat, 84))),
        255,
        cv2.THRESH_BINARY,
    )

    combined = cv2.bitwise_and(thresholded, blackhat_mask)
    combined = cv2.bitwise_and(combined, paper_mask)
    combined = cv2.medianBlur(combined, 3)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    candidates = _refine_candidates(_extract_dot_candidates(combined, filtered))
    rows = _bucket_rows(candidates)
    text, confidence, roi_boxes = _decode_rows(rows)
    mapped_boxes = [[x + roi_left, y + roi_top, w, h] for x, y, w, h in roi_boxes]
    debug_image = _encode_debug_image(combined)
    return text, confidence, mapped_boxes, debug_image


def _process_decoded_frame(frame: np.ndarray) -> Dict[str, object]:
    text, confidence, boxes, debug_image = process_braille_frame(frame)
    return {
        "text": text,
        "confidence": confidence,
        "boxes": boxes,
        "debug_image": debug_image,
    }


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process-braille")
async def process_braille_base64(payload: Base64ImagePayload = Body(...)) -> Dict[str, object]:
    frame = _decode_base64_image(payload.image)
    return _process_decoded_frame(frame)


@app.post("/api/process-braille/upload")
async def process_braille_upload(file: UploadFile = File(...)) -> Dict[str, object]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="No uploaded image data was provided.")

    frame = _decode_image(raw)
    return _process_decoded_frame(frame)


@app.post("/api/process-braille/capture")
async def process_braille_capture(payload: Base64ImagePayload = Body(...)) -> Dict[str, object]:
    frame = _decode_base64_image(payload.image)
    return _process_decoded_frame(frame)


@app.post("/api/v1/process-frame")
async def process_frame(file: UploadFile = File(...)) -> Dict[str, object]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Received empty frame.")

    frame = _decode_image(raw)
    return _process_decoded_frame(frame)
