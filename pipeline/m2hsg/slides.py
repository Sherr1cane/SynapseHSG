"""Slide processing: image dimension parsing, grid region generation, and
slide + vision-index construction for a single session."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import SLIDE_EXTS, SessionRecord, _safe_float

# ---------------------------------------------------------------------------
# Image dimension helpers (no Pillow required)
# ---------------------------------------------------------------------------


def parse_png_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read PNG width/height from the IHDR chunk (first 24 bytes)."""
    try:
        data = path.read_bytes()[:24]
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return None, None
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        return w, h
    except Exception:
        return None, None


def parse_jpeg_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read JPEG width/height by scanning SOF markers."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
            return None, None
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            i += 2
            if marker in (0xD8, 0xD9):
                continue
            if i + 1 >= len(data):
                break
            seg_len = (data[i] << 8) + data[i + 1]
            if seg_len < 2:
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                if i + 7 >= len(data):
                    break
                h = (data[i + 3] << 8) + data[i + 4]
                w = (data[i + 5] << 8) + data[i + 6]
                return w, h
            i += seg_len
    except Exception:
        return None, None
    return None, None


def image_dimensions(path: Path) -> Tuple[int, int]:
    """Return (width, height) for a PNG or JPEG slide image.

    Falls back to (1000, 1000) when the header cannot be parsed.
    """
    suffix = path.suffix.lower()
    if suffix == ".png":
        w, h = parse_png_dimensions(path)
    elif suffix in (".jpg", ".jpeg"):
        w, h = parse_jpeg_dimensions(path)
    else:
        w, h = None, None
    return (w or 1000, h or 1000)


# ---------------------------------------------------------------------------
# Grid region generation
# ---------------------------------------------------------------------------


def make_grid_regions(width: int, height: int) -> List[List[int]]:
    """Return a list of [x0, y0, x1, y1] bounding boxes.

    The first region covers the full image; the remaining four are quadrants.
    """
    return [
        [0, 0, width, height],
        [0, 0, width // 2, height // 2],
        [width // 2, 0, width, height // 2],
        [0, height // 2, width // 2, height],
        [width // 2, height // 2, width, height],
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_slides_and_vision_index(
    rec: SessionRecord,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build the *slides_structured* and *vision_index* dicts for *rec*.

    Returns
    -------
    slides_structured : dict
        Keys: ``sample_id``, ``slides``, ``visual_regions``.
    vision_index : dict
        Keys: ``sample_id``, ``regions``.
    """
    slide_dir = rec.path / "slides"
    images: List[Path] = []
    if slide_dir.exists():
        images = sorted(
            [p for p in slide_dir.iterdir() if p.suffix.lower() in SLIDE_EXTS]
        )

    slide_entries = rec.metadata.get("slides_data", {}).get("slides", [])
    times_sec = [_safe_float(x.get("time", 0), 0.0) / 1000.0 for x in slide_entries]

    slides: List[Dict[str, Any]] = []
    visual_regions: List[Dict[str, Any]] = []
    vision_entries: List[Dict[str, Any]] = []

    for i, img in enumerate(images):
        slide_id = f"{rec.sample_id}/slide/{i:04d}"
        start = round(times_sec[i], 3) if i < len(times_sec) else None
        end = round(times_sec[i + 1], 3) if i + 1 < len(times_sec) else None
        w, h = image_dimensions(img)

        region_ids: List[str] = []
        for ridx, bbox in enumerate(make_grid_regions(w, h)):
            region_id = f"{slide_id}/region/{ridx:02d}"
            region_ids.append(region_id)
            visual_regions.append(
                {
                    "region_id": region_id,
                    "slide_id": slide_id,
                    "bbox": bbox,
                    "region_type": "full" if ridx == 0 else "quadrant",
                    "source_image": str(img.name),
                    "source_subdir": "slides",
                }
            )
            vision_entries.append(
                {
                    "region_id": region_id,
                    "source_type": "slide",
                    "source_ref": slide_id,
                    "slide_id": slide_id,
                    "bbox": bbox,
                    "region_type": "full" if ridx == 0 else "quadrant",
                    "source_image": str(img.name),
                    "source_subdir": "slides",
                }
            )

        slides.append(
            {
                "slide_id": slide_id,
                "index": i,
                "image_file": str(img.name),
                "start_time": start,
                "end_time": end,
                "image_width": w,
                "image_height": h,
                "region_ids": region_ids,
            }
        )

    slides_structured: Dict[str, Any] = {
        "sample_id": rec.sample_id,
        "slides": slides,
        "visual_regions": visual_regions,
    }
    vision_index: Dict[str, Any] = {
        "sample_id": rec.sample_id,
        "regions": vision_entries,
    }
    return slides_structured, vision_index
